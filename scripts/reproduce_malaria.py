"""Reproduce Experiment 1 (Malaria CNN distillation) end-to-end.

Trains an ensemble of weak teachers, builds the ASR, and runs the SPRKD
student training loop. Produces:

* ``checkpoints/teacher_{i}.pth``  - one per teacher in the ensemble.
* ``checkpoints/asr.pth``          - aggregated Approximated Saddle Region.
* ``checkpoints/sprkd_student.pth``
* ``checkpoints/control_student.pth``
* ``checkpoints/rkd_student.pth``
* ``checkpoints/results.json``     - top-1 valid accuracies + Hessian trace.
"""

from __future__ import annotations

import argparse
import json
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
from sprkd.training import (
    TrainingHistory,
    train_control,
    train_response_kd,
    train_student,
    train_teacher,
)
from sprkd.utils import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=find_default_root())
    p.add_argument("--n-teachers", type=int, default=3)
    p.add_argument("--teacher-epochs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--skip-rkd", action="store_true")
    p.add_argument("--skip-control", action="store_true")
    return p.parse_args()


def _save_history(path: Path, history: TrainingHistory):
    torch.save(history.to_dict(), path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MalariaDataConfig(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, valid_loader, _ = make_dataloaders(cfg)

    repos: list[list] = []
    teacher_models = []
    for i in range(args.n_teachers):
        set_seed(args.seed + i)
        print(f"[reproduce] training teacher {i + 1}/{args.n_teachers}")
        teacher = MalariaTeacherCNN().to(device)
        sprkd_t, history = train_teacher(
            teacher,
            train_loader,
            valid_loader,
            loss_fn=nn.CrossEntropyLoss(),
            n_epochs=args.teacher_epochs,
            saddle_steps=1,
            device=device,
        )
        if len(sprkd_t.saddle_repository) == 0:
            sprkd_t.saddle_repository.append(
                list(teacher.parameters()), loss=float(history.train_losses[-1])
            )
        torch.save(
            {
                "model_state_dict": teacher.state_dict(),
                "saddle_snapshots": sprkd_t.saddle_repository.snapshots,
                "saddle_losses": sprkd_t.saddle_repository.losses,
                "history": history.to_dict(),
            },
            args.out_dir / f"teacher_{i}.pth",
        )
        repos.append(sprkd_t.saddle_repository.snapshots)
        teacher_models.append(teacher)

    print("[reproduce] aggregating ASR")
    asr = aggregate_asr(repos)
    torch.save(asr, args.out_dir / "asr.pth")

    set_seed(args.seed)
    student = MalariaStudentCNN().to(device)
    inject_state_list(student, asr, teacher=teacher_models[0])
    targets = [p.detach().clone() for p in student.parameters()]

    print("[reproduce] SPRKD student training")
    _, sprkd_history = train_student(
        student,
        train_loader,
        valid_loader,
        loss_fn=nn.CrossEntropyLoss(),
        teacher_saddle_points=targets,
        n_epochs=args.epochs,
        device=device,
    )
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "history": sprkd_history.to_dict(),
        },
        args.out_dir / "sprkd_student.pth",
    )

    results = {
        "sprkd_best_val_acc": sprkd_history.best_valid_acc(),
        "teacher_best_val_acc": [],
    }

    if not args.skip_control:
        set_seed(args.seed)
        control = MalariaStudentCNN().to(device)
        ctrl_history = train_control(
            control,
            train_loader,
            valid_loader,
            loss_fn=nn.CrossEntropyLoss(),
            n_epochs=args.epochs,
            device=device,
        )
        torch.save(
            {
                "model_state_dict": control.state_dict(),
                "history": ctrl_history.to_dict(),
            },
            args.out_dir / "control_student.pth",
        )
        results["control_best_val_acc"] = ctrl_history.best_valid_acc()

    if not args.skip_rkd:
        set_seed(args.seed)
        rkd_student = MalariaStudentCNN().to(device)
        rkd_history = train_response_kd(
            rkd_student,
            teacher_models[0],
            train_loader,
            valid_loader,
            n_epochs=args.epochs,
            device=device,
        )
        torch.save(
            {
                "model_state_dict": rkd_student.state_dict(),
                "history": rkd_history.to_dict(),
            },
            args.out_dir / "rkd_student.pth",
        )
        results["rkd_best_val_acc"] = rkd_history.best_valid_acc()

    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
