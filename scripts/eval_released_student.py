"""Evaluate the released SPRKD student checkpoint on the held-out test set.

Two evaluation paths are supported:

* ``--testset TESTSET.pth`` (default, fast) - the 100-sample held-out tensor
  pair shipped under git-lfs. This is the same set used in the paper for
  per-class evaluation.
* ``--use-valid-split`` - rebuild the validation split from ``cell_images/``
  and evaluate on the full (~6,890 sample) split.

Both paths print a JSON payload with validation accuracy and loss. The
relative ordering reported in the paper (SPRKD > Control > RKD) is
reproduced on the held-out test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from sprkd import load_legacy_student
from sprkd.data import MalariaDataConfig, find_default_root, make_dataloaders
from sprkd.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("MODELS") / "SPRKD_MALARIA.pth",
    )
    p.add_argument(
        "--testset",
        type=Path,
        default=Path("TESTSET.pth"),
    )
    p.add_argument("--use-valid-split", action="store_true")
    p.add_argument("--data-root", type=Path, default=find_default_root())
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _eval_on_tensors(model: nn.Module, xs: torch.Tensor, ys: torch.Tensor, device, loss_fn):
    xs, ys = xs.to(device), ys.to(device)
    with torch.no_grad():
        logits = model(xs)
        loss = loss_fn(logits, ys).item()
        preds = torch.argmax(logits, dim=1)
        acc = 100.0 * (preds == ys).float().mean().item()
    return acc, loss, ys.numel()


def _eval_on_loader(model: nn.Module, loader, device, loss_fn):
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)
    return (
        100.0 * correct / max(1, total),
        total_loss / max(1, total),
        total,
    )


def main() -> None:
    args = parse_args()
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()
    student = load_legacy_student(args.checkpoint).to(device)
    student.eval()

    if not args.use_valid_split and args.testset.is_file():
        ts = torch.load(args.testset, map_location="cpu", weights_only=False)
        xs, ys = ts[0], ts[1]
        acc, loss, n = _eval_on_tensors(student, xs, ys, device, loss_fn)
        source = str(args.testset)
    else:
        cfg = MalariaDataConfig(
            root=args.data_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        _, valid_loader, _ = make_dataloaders(cfg)
        acc, loss, n = _eval_on_loader(student, valid_loader, device, loss_fn)
        source = str(args.data_root) + " (valid split)"

    payload = {
        "checkpoint": str(args.checkpoint),
        "evaluation_source": source,
        "device": str(device),
        "val_accuracy": round(acc, 4),
        "val_loss": round(loss, 6),
        "n_samples": n,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
