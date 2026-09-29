"""Shared utilities: seeding, device, logging, evaluation, LR schedule, KD loss."""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        if prefer == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> Dict[str, float]:
    model.eval()
    correct1 = correct5 = n = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == y).sum().item()
        correct5 += (top5 == y[:, None]).any(dim=1).sum().item()
        n += y.numel()
    return {"top1": 100.0 * correct1 / n, "top5": 100.0 * correct5 / n, "loss": loss_sum / n, "n": n}


def kd_loss(student_logits, teacher_logits, T: float = 4.0) -> torch.Tensor:
    """Hinton KD: KL(p_T^teacher || p_T^student) * T^2 (batchmean)."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


def atomic_save(obj, path) -> None:
    """torch.save to a temp file in the same directory, then atomic rename. A kill or preemption
    mid-write leaves the previous file intact instead of a truncated one (a truncated resume.pt
    made a restarted run crash on 2026-09-28 in local testing)."""
    import os
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def safe_load(path, **kw):
    """torch.load that returns None (with a warning) for a missing or unreadable file."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return torch.load(path, **kw)
    except Exception as e:  # truncated or corrupt file
        print(json.dumps({"warning": f"could not load {path.name}: {type(e).__name__}; starting this phase fresh"}), flush=True)
        return None


def make_sgd(params, lr: float, momentum: float = 0.9, weight_decay: float = 5e-4):
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)


def make_multistep(optimizer, milestones: List[int], gamma: float = 0.1):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


@dataclass
class RunLogger:
    """Per-run CSV + JSON logger. One row per epoch; final summary JSON."""
    run_dir: Path
    config: dict
    rows: List[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.time)

    def __post_init__(self):
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)
        self.csv_path = self.run_dir / "epochs.csv"

    def log_epoch(self, **kv):
        kv = {"elapsed_s": round(time.time() - self.t0, 1), **kv}
        self.rows.append(kv)
        # Rows from different phases carry different keys (teacher / inject / student), so rewrite the
        # CSV with the union of columns each time (rows are few) and keep a JSONL copy as the source of truth.
        cols = list(dict.fromkeys(k for r in self.rows for k in r))
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(self.rows)
        with open(self.run_dir / "epochs.jsonl", "a") as f:
            f.write(json.dumps(kv, default=str) + "\n")
        print(json.dumps(kv), flush=True)
        total = self.config.get("epochs") or self.config.get("student_epochs")
        if "epoch" in kv and total:  # heartbeat the Modal driver reads to project wall-clock
            print(json.dumps({"progress": int(kv["epoch"]), "of": int(total), "elapsed_s": kv["elapsed_s"]}), flush=True)

    def finish(self, **extra):
        best = max((r.get("test_top1", float("-inf")) for r in self.rows), default=float("nan"))
        last = self.rows[-1] if self.rows else {}
        summary = {
            "config": self.config,
            "best_test_top1": best,
            "final_test_top1": last.get("test_top1"),
            "final_test_top5": last.get("test_top5"),
            "final_test_loss": last.get("test_loss"),
            "epochs_run": len(self.rows),
            "wall_clock_s": round(time.time() - self.t0, 1),
            **extra,
        }
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("SUMMARY " + json.dumps(summary, default=str), flush=True)
        return summary
