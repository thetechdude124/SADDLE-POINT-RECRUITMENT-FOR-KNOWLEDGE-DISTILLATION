"""End-to-end quickstart for the SPRKD paper's Experiment 1 (Malaria CNN).

Run from the repo root:

    python examples/quickstart.py --epochs 2 --teacher-epochs 2

This is the *minimum* number of moving parts to demonstrate the SPRKD
pipeline:

1. Train a weak teacher on the NIH/NLM malaria split with saddle tracking.
2. Aggregate the recorded saddles into an Approximated Saddle Region (ASR).
3. Inject the ASR into a 4x-smaller student via ``inject_state_list`` (TLI).
4. Run the SPRKD student training loop (Transformation Matrix -> NHE -> PGD).
5. Print best validation accuracy.

The script uses the bundled ``cell_images`` directory by default; pass
``--data-root`` to point at a different copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from sprkd import (
    MalariaStudentCNN,
    MalariaTeacherCNN,
    aggregate_asr,
    set_seed,
)
from sprkd.tli import inject_state_list
from sprkd.data import MalariaDataConfig, find_default_root, make_dataloaders
from sprkd.training import train_student, train_teacher
from sprkd.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=find_default_root())
    p.add_argument("--teacher-epochs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Cap on training batches per epoch (for fast smoke tests).",
    )
    return p.parse_args()


class _LimitedLoader:
    """Wrap a DataLoader to expose only the first N batches per epoch."""

    def __init__(self, loader, n):
        self._loader = loader
        self._n = n

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                break
            yield batch

    def __len__(self):
        try:
            return min(self._n, len(self._loader))
        except TypeError:
            return self._n


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"[quickstart] device={device}")

    cfg = MalariaDataConfig(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, valid_loader, full = make_dataloaders(cfg)
    print(f"[quickstart] dataset size: {len(full)} (train+valid)")

    if args.steps_per_epoch is not None:
        train_loader = _LimitedLoader(train_loader, args.steps_per_epoch)

    # 1. teacher with saddle tracking
    teacher = MalariaTeacherCNN().to(device)
    print("[quickstart] training teacher...")
    sprkd_t, t_history = train_teacher(
        teacher,
        train_loader,
        valid_loader,
        loss_fn=nn.CrossEntropyLoss(),
        n_epochs=args.teacher_epochs,
        saddle_steps=1,
        device=device,
    )
    print(
        f"[quickstart] teacher recorded {len(sprkd_t.saddle_repository)} saddle "
        f"snapshots, best val acc {t_history.best_valid_acc():.2f}%"
    )

    if len(sprkd_t.saddle_repository) == 0:
        print(
            "[quickstart] WARNING: no saddles recorded - falling back to current "
            "teacher params as the ASR (this is the 'control' configuration)."
        )
        sprkd_t.saddle_repository.append(
            list(teacher.parameters()), loss=float(t_history.train_losses[-1])
        )

    # 2. ASR
    asr = aggregate_asr([sprkd_t.saddle_repository.snapshots])
    print(f"[quickstart] ASR has {len(asr)} layers")

    # 3. student + TLI: load ASR into teacher shapes, then inject into student
    student = MalariaStudentCNN().to(device)
    inject_state_list(student, asr, teacher=teacher)
    targets = [p.detach().clone() for p in student.parameters()]

    # 4. SPRKD student loop
    print("[quickstart] training SPRKD student...")
    sprkd_s, s_history = train_student(
        student,
        train_loader,
        valid_loader,
        loss_fn=nn.CrossEntropyLoss(),
        teacher_saddle_points=targets,
        n_epochs=args.epochs,
        device=device,
        sprkd_kwargs={"max_nhe_steps": 50},
    )

    # 5. report
    print("[quickstart] DONE")
    print(f"  teacher val acc:  {t_history.best_valid_acc():.2f}%")
    print(f"  SPRKD student:    {s_history.best_valid_acc():.2f}%")


if __name__ == "__main__":
    main()
