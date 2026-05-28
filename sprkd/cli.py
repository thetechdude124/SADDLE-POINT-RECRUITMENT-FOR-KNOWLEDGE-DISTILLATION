"""Lightweight command-line interface for the SPRKD reproduction pipeline.

Run via:

.. code-block:: bash

    sprkd info
    sprkd train-teacher --epochs 2 --output checkpoints/teacher.pth
    sprkd build-asr   --teachers TEACHER_*.pth --output ASR.pth
    sprkd train-student --asr ASR.pth --epochs 10 --output student.pth
    sprkd eval        --student student.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from sprkd import __version__
from sprkd.data import MalariaDataConfig, find_default_root, make_dataloaders
from sprkd.models import (
    MalariaStudentCNN,
    MalariaTeacherCNN,
    count_parameters,
)
from sprkd.saddle import aggregate_asr
from sprkd.tli import inject_state_list
from sprkd.training import (
    train_control,
    train_response_kd,
    train_student,
    train_teacher,
)
from sprkd.utils import get_device, set_seed


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def _cmd_info(args: argparse.Namespace) -> int:
    payload = {
        "sprkd_version": __version__,
        "torch_version": torch.__version__,
        "device": str(get_device()),
        "models": {
            "MalariaTeacherCNN": count_parameters(MalariaTeacherCNN()),
            "MalariaStudentCNN": count_parameters(MalariaStudentCNN()),
        },
        "default_dataset_root": str(find_default_root()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _build_dataloaders(args: argparse.Namespace):
    root = Path(args.data_root) if args.data_root else find_default_root()
    cfg = MalariaDataConfig(
        root=root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    return make_dataloaders(cfg)


def _cmd_train_teacher(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = get_device()
    train_loader, valid_loader, _ = _build_dataloaders(args)

    model = MalariaTeacherCNN().to(device)
    sprkd, history = train_teacher(
        model,
        train_loader,
        valid_loader,
        loss_fn=nn.CrossEntropyLoss(),
        n_epochs=args.epochs,
        lr=args.lr,
        saddle_steps=args.saddle_steps,
        device=device,
        progress=not args.quiet,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "saddle_snapshots": sprkd.saddle_repository.snapshots,
            "saddle_losses": sprkd.saddle_repository.losses,
            "history": history.to_dict(),
        },
        out,
    )
    print(f"Saved teacher + saddles to {out} (recorded {len(sprkd.saddle_repository)} saddles)")
    return 0


def _cmd_build_asr(args: argparse.Namespace) -> int:
    repos = []
    for path in args.teachers:
        ckpt = torch.load(path, map_location="cpu")
        repos.append(ckpt["saddle_snapshots"])
    asr = aggregate_asr(repos)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(asr, out)
    print(f"Saved ASR ({len(asr)} tensors) to {out}")
    return 0


def _cmd_train_student(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = get_device()
    train_loader, valid_loader, _ = _build_dataloaders(args)

    student = MalariaStudentCNN().to(device)
    asr = torch.load(args.asr, map_location=device)

    teacher = MalariaTeacherCNN().to(device)
    inject_state_list(student, asr, teacher=teacher)

    teacher_saddle_points = [p.detach().clone() for p in student.parameters()]

    sprkd, history = train_student(
        student,
        train_loader,
        valid_loader,
        loss_fn=nn.CrossEntropyLoss(),
        teacher_saddle_points=teacher_saddle_points,
        n_epochs=args.epochs,
        lr=args.lr,
        device=device,
        progress=not args.quiet,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "history": history.to_dict(),
            "best_valid_acc": history.best_valid_acc(),
        },
        out,
    )
    print(f"Saved student to {out} (best val acc {history.best_valid_acc():.2f}%)")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    device = get_device()
    train_loader, valid_loader, _ = _build_dataloaders(args)
    ckpt = torch.load(args.student, map_location=device)

    student = MalariaStudentCNN().to(device)
    student.load_state_dict(ckpt["model_state_dict"])
    student.eval()

    loss_fn = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            logits = student(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total += y.size(0)
    print(
        json.dumps(
            {
                "val_accuracy": round(100.0 * correct / max(1, total), 4),
                "val_loss": round(total_loss / max(1, total), 6),
                "n_samples": total,
            },
            indent=2,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sprkd", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Print package + environment info").set_defaults(
        func=_cmd_info
    )

    p_t = sub.add_parser("train-teacher", help="Train a single teacher with saddle tracking")
    _add_data_args(p_t)
    p_t.add_argument("--epochs", type=int, default=2)
    p_t.add_argument("--lr", type=float, default=1e-3)
    p_t.add_argument("--saddle-steps", type=int, default=1)
    p_t.add_argument("--output", type=str, required=True)
    p_t.set_defaults(func=_cmd_train_teacher)

    p_a = sub.add_parser("build-asr", help="Aggregate teacher saddles into an ASR")
    p_a.add_argument("--teachers", type=str, nargs="+", required=True)
    p_a.add_argument("--output", type=str, required=True)
    p_a.set_defaults(func=_cmd_build_asr)

    p_s = sub.add_parser("train-student", help="Train a student via SPRKD")
    _add_data_args(p_s)
    p_s.add_argument("--asr", type=str, required=True)
    p_s.add_argument("--epochs", type=int, default=10)
    p_s.add_argument("--lr", type=float, default=1e-3)
    p_s.add_argument("--output", type=str, required=True)
    p_s.set_defaults(func=_cmd_train_student)

    p_e = sub.add_parser("eval", help="Evaluate a saved student on the malaria valid split")
    _add_data_args(p_e)
    p_e.add_argument("--student", type=str, required=True)
    p_e.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
