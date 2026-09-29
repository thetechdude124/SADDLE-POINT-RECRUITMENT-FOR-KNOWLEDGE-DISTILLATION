"""Scratch / vanilla-KD training on CIFAR-100 with the CRD recipe.

Examples
--------
# scratch student, 240 epochs, seed 0
python train.py --mode scratch --model resnet8x4 --seed 0

# vanilla KD (Hinton, T=4, alpha=0.9 KD / 0.1 CE) from a saved teacher
python train.py --mode kd --model resnet8x4 --teacher resnet32x4 \
    --teacher-ckpt checkpoints/resnet32x4_scratch_s0/best.pt

Recipe (Tian et al. 2020): SGD momentum 0.9, wd 5e-4, batch 64, lr 0.05
(0.01 for MobileNet/ShuffleNet), 240 epochs, lr x0.1 at 150/180/210.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RunLogger, atomic_save, evaluate, get_device, kd_loss, make_multistep, make_sgd, safe_load, set_seed
from data import get_loaders
from models import build_model, count_params

HERE = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["scratch", "kd"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--teacher", default=None)
    p.add_argument("--teacher-ckpt", default=None)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--milestones", type=int, nargs="*", default=[150, 180, 210])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--kd-T", type=float, default=4.0)
    p.add_argument("--kd-alpha", type=float, default=0.9, help="weight on KD term; CE gets 1-alpha")
    p.add_argument("--dataset", choices=["cifar100", "tinyimagenet", "imagenet100"], default="cifar100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--subset", type=int, default=None, help="use N train images (smoke tests)")
    p.add_argument("--data-root", default=str(HERE / "data"))
    p.add_argument("--out-root", default=str(HERE / "results" / "runs"))
    p.add_argument("--tag", default="")
    p.add_argument("--run-name", default=None, help="override the auto-generated run directory name")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from <run_dir>/resume.pt if present; skip if summary.json exists")
    p.add_argument("--ckpt-fractions", type=float, nargs="*", default=None, help="save ck_fXXX.pt at these fractions of the total steps (model only)")
    return p.parse_args()


def remap_crd_keys(sd: dict, name: str) -> dict:
    """Map RepDistiller (CRD) checkpoint keys onto this harness's module names."""
    out = {}
    for k, v in sd.items():
        k2 = k
        if name.startswith("wrn"):
            k2 = re.sub(r"^(block[123])\.layer\.", r"\1.", k2)   # WRN: blockN.layer.i -> blockN.i
            k2 = re.sub(r"^bn1\.", "bn.", k2)                       # WRN final BN
            k2 = k2.replace(".convShortcut.", ".shortcut.")           # WRN 1x1 shortcut
        elif name.startswith("vgg"):
            k2 = re.sub(r"^block(\d)\.", r"stages.\1.", k2)       # VGG: blockK -> stages.K
        out[k2] = v
    return out


def load_teacher(name: str, ckpt: str, device, num_classes: int = 100):
    t = build_model(name, num_classes).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = remap_crd_keys(sd, name)
    missing, unexpected = t.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if bad_missing or unexpected:
        raise RuntimeError(f"teacher load mismatch: missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def main():
    args = parse()
    set_seed(args.seed)
    device = get_device(args.device)
    run_name = f"{args.model}_{args.mode}" + ("" if args.dataset == "cifar100" else f"_{args.dataset}") + (f"_from_{args.teacher}" if args.mode == "kd" else "") + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_name = args.run_name or run_name
    run_dir = Path(args.out_root) / run_name
    if args.resume and (run_dir / "summary.json").is_file():
        print(f"skip {run_name}: summary.json exists", flush=True); return
    logger = RunLogger(run_dir, {**vars(args), "run_name": run_name, "device": str(device)})

    train_loader, test_loader, num_classes = get_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers, args.seed, args.subset, download=True)
    model = build_model(args.model, num_classes).to(device)
    print(f"model={args.model} params={count_params(model)/1e6:.2f}M device={device}", flush=True)

    teacher = None
    if args.mode == "kd":
        assert args.teacher and args.teacher_ckpt, "--teacher and --teacher-ckpt required for kd"
        teacher = load_teacher(args.teacher, args.teacher_ckpt, device, num_classes)
        t_eval = evaluate(teacher, test_loader, device)
        print(f"teacher={args.teacher} test_top1={t_eval['top1']:.2f}", flush=True)
        logger.config["teacher_test_top1"] = t_eval["top1"]

    opt = make_sgd(model.parameters(), args.lr, weight_decay=args.wd)
    sched = make_multistep(opt, args.milestones)
    best = -1.0
    start_epoch = 1
    resume_path = run_dir / "resume.pt"
    ck = safe_load(resume_path, map_location=device, weights_only=False) if args.resume else None
    if ck is not None:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, start_epoch = ck["best"], ck["epoch"] + 1
        logger.rows = ck.get("rows", [])
        print(f"resumed {run_name} at epoch {start_epoch}", flush=True)
    total_steps = args.epochs * len(train_loader)
    frac_steps = {max(1, round(f * total_steps)): f for f in (args.ckpt_fractions or [])}
    global_step = (start_epoch - 1) * len(train_loader)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum = n = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = (1 - args.kd_alpha) * ce + args.kd_alpha * kd_loss(logits, t_logits, args.kd_T)
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            global_step += 1
            if global_step in frac_steps:
                atomic_save({"model": model.state_dict(), "step": global_step, "fraction": frac_steps[global_step]}, run_dir / f"ck_f{int(round(frac_steps[global_step] * 100)):03d}.pt")
            loss_sum += loss.item() * y.numel(); n += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        ev = evaluate(model, test_loader, device)
        logger.log_epoch(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=loss_sum / n, train_top1=100 * correct / n,
                         test_top1=ev["top1"], test_top5=ev["top5"], test_loss=ev["loss"])
        if ev["top1"] > best:
            best = ev["top1"]
            if args.save_ckpt:
                atomic_save({"model": model.state_dict(), "epoch": epoch, "test_top1": best}, run_dir / "best.pt")
        if args.resume:
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best,
                        "epoch": epoch, "rows": logger.rows}, resume_path)
    if args.save_ckpt:
        atomic_save({"model": model.state_dict(), "epoch": args.epochs, "test_top1": ev["top1"]}, run_dir / "last.pt")
    logger.finish()


if __name__ == "__main__":
    main()
