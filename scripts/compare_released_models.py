"""Side-by-side comparison of the SPRKD / Control / RKD students.

Loads the released checkpoints from ``MODELS/`` and evaluates them on
``TESTSET.pth``. Reproduces the paper's relative ordering (SPRKD > Control >
RKD) without any retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from sprkd import load_legacy_student
from sprkd.utils import get_device


DEFAULT_MODELS = {
    "SPRKD":   Path("MODELS") / "SPRKD_MALARIA.pth",
    "Control": Path("MODELS") / "CONTROL_MALARIA.pth",
    "RKD":     Path("MODELS") / "RKD_MALARIA_STUDENT.pth",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--testset", type=Path, default=Path("TESTSET.pth"))
    return p.parse_args()


def _eval(model: nn.Module, xs, ys, device, loss_fn):
    xs, ys = xs.to(device), ys.to(device)
    with torch.no_grad():
        logits = model(xs)
        loss = loss_fn(logits, ys).item()
        preds = torch.argmax(logits, dim=1)
        acc = 100.0 * (preds == ys).float().mean().item()
    return acc, loss


def main() -> None:
    args = parse_args()
    device = get_device()
    loss_fn = nn.CrossEntropyLoss()

    ts = torch.load(args.testset, map_location="cpu", weights_only=False)
    xs, ys = ts[0], ts[1]

    rows = []
    for label, path in DEFAULT_MODELS.items():
        if not path.is_file():
            print(f"[skip] {label}: checkpoint missing at {path}")
            continue
        try:
            model = load_legacy_student(path).to(device)
        except Exception as e:
            print(f"[skip] {label}: failed to load -- {e}")
            continue
        model.eval()
        acc, loss = _eval(model, xs, ys, device, loss_fn)
        rows.append({"model": label, "accuracy": round(acc, 4), "loss": round(loss, 6)})

    rows.sort(key=lambda r: -r["accuracy"])
    print(json.dumps({"n_samples": int(len(ys)), "device": str(device), "results": rows}, indent=2))


if __name__ == "__main__":
    main()
