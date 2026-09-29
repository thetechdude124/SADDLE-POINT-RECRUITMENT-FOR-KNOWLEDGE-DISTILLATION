"""SPRKD on CIFAR-100 CRD pairs, driven by the released `sprkd` package.

Pipeline (mirrors the paper, Section 3):
  1. train K weak teachers for `--teacher-epochs` with SPRKD teacher-mode saddle tracking
     (PyHessian top-k eigenvalues every `--saddle-steps` steps on a fixed probe batch);
  2. aggregate the recorded snapshots into an ASR (`sprkd.saddle.aggregate_asr`);
  3. inject the ASR into the student by state-dict key matching + center crop
     (`sprkd.tli.inject_state_list`);
  4. train the student with the SPRKD optimizer in student mode wrapping the CRD SGD recipe.

Ablation switches (each disables one component):
  --no-tm        skip the iterative "transformation matrix" phase (direct init at ASR)
  --no-nhe       max_nhe_steps=0
  --no-pgd       pgd_perturb_variance=0 (and effectively no perturbation)
  --asr-mode     {saddle, last, random_snapshot, swa}: what the ASR is built from
  --init-only    inject the ASR and then train with plain SGD (no SPRKD optimizer at all)
  --teachers K   number of weak teachers

Everything is logged like train.py so results/aggregate.py can read it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RunLogger, atomic_save, evaluate, get_device, make_multistep, make_sgd, safe_load, set_seed
from data import get_loaders
from models import build_model, count_params

HERE = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="student")
    p.add_argument("--teacher", required=True, help="teacher architecture (weak, trained here)")
    p.add_argument("--teachers", type=int, default=1, help="ensemble size")
    p.add_argument("--teacher-epochs", type=int, default=2, help="0 = do not train (use --teacher-ckpt weights as-is)")
    p.add_argument("--teacher-ckpt", default=None, help="load teacher weights (CRD checkpoint or a run's best.pt) before/instead of training")
    p.add_argument("--teacher-lr", type=float, default=0.05)
    p.add_argument("--saddle-steps", type=int, default=50)
    p.add_argument("--saddle-limit", type=int, default=None, help="stop recording after N steps")
    p.add_argument("--n-top-eigs", type=int, default=4)
    p.add_argument("--saddle-rule", choices=["extreme", "magnitude", "ratio", "both"], default="extreme")
    p.add_argument("--refine", action="store_true", help="refine candidates to stationary points; record only verified saddles")
    p.add_argument("--refine-grad-tol", type=float, default=1e-2)
    p.add_argument("--nhe-direction", choices=["lambda_min", "topk"], default="lambda_min")
    p.add_argument("--hessian-batch", type=int, default=64)
    p.add_argument("--asr-mode", choices=["saddle", "last", "random_snapshot", "swa"], default="saddle")
    p.add_argument("--init-only", action="store_true")
    p.add_argument("--no-tm", action="store_true")
    p.add_argument("--no-nhe", action="store_true")
    p.add_argument("--no-pgd", action="store_true")
    p.add_argument("--epsilon", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=240)
    p.add_argument("--milestones", type=int, nargs="*", default=[150, 180, 210])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dataset", choices=["cifar100", "tinyimagenet", "imagenet100"], default="cifar100")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--data-root", default=str(HERE / "data"))
    p.add_argument("--out-root", default=str(HERE / "results" / "runs"))
    p.add_argument("--tag", default="")
    p.add_argument("--run-name", default=None, help="override the auto-generated run directory name")
    p.add_argument("--save-ckpt", action="store_true")
    p.add_argument("--resume", action="store_true", help="cache the teacher phase and resume the student from <run_dir>/resume.pt; skip if summary.json exists")
    p.add_argument("--teacher-cache", default=None, help="shared teacher-phase cache file (same teacher/seed/epochs/saddle settings across arms)")
    return p.parse_args()


def _probe_batch(loader, n: int, device):
    x, y = next(iter(loader))
    return x[:n].to(device), y[:n].to(device)


def train_weak_teacher(args, k: int, train_loader, test_loader, device, logger):
    from sprkd.optimizer import SPRKD
    from sprkd.saddle import SaddleCriterion

    set_seed(args.seed * 100 + k)
    teacher = build_model(args.teacher, args.num_classes).to(device)
    if args.teacher_ckpt:
        from train import load_teacher
        teacher = load_teacher(args.teacher, args.teacher_ckpt, device, args.num_classes)
        for p_ in teacher.parameters():
            p_.requires_grad_(True)
        ev = evaluate(teacher, test_loader, device)
        logger.log_epoch(phase=f"teacher{k}", epoch=0, test_top1=ev["top1"], test_top5=ev["top5"], test_loss=ev["loss"], n_saddles=0, loaded_ckpt=args.teacher_ckpt)
    base = make_sgd(teacher.parameters(), args.teacher_lr, weight_decay=args.wd)
    opt = SPRKD(
        teacher.parameters(), base_optimizer=base, loss_fn=nn.CrossEntropyLoss(),
        is_teacher=True, saddle_steps=args.saddle_steps, saddle_step_limit=args.saddle_limit,
        n_top_eigs=args.n_top_eigs, saddle_criterion=SaddleCriterion(rule=args.saddle_rule),
        saddle_refine=args.refine, refine_grad_tol=args.refine_grad_tol, refine_max_steps=30,
    )
    probe = _probe_batch(train_loader, args.hessian_batch, device)
    snapshots_all = []  # every checked step, for the random_snapshot / swa ablations
    t0 = time.time()
    step = 0
    for ep in range(1, args.teacher_epochs + 1):
        teacher.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(teacher(x), y)
            loss.backward()
            n_before = len(opt.saddle_repository)
            gn = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in teacher.parameters() if p.grad is not None)).item()
            opt.step(model=teacher, current_loss=loss.detach(), data_batch=probe)
            teacher.train()  # PyHessian leaves the model in eval mode; undo that
            step += 1
            if step % args.saddle_steps == 0 and args.asr_mode in ("random_snapshot", "swa"):
                snapshots_all.append([p.detach().clone().cpu() for p in teacher.parameters()])
            if len(opt.saddle_repository) > n_before:
                print(json.dumps({"event": "saddle_recorded", "teacher": k, "step": step, "loss": float(loss), "grad_norm": gn}), flush=True)
        ev = evaluate(teacher, test_loader, device)
        logger.log_epoch(phase=f"teacher{k}", epoch=ep, test_top1=ev["top1"], test_top5=ev["top5"], test_loss=ev["loss"],
                         n_saddles=len(opt.saddle_repository), teacher_time_s=round(time.time() - t0, 1))
    return teacher, opt.saddle_repository, snapshots_all


def build_asr(args, repos, all_snaps, teachers):
    from sprkd.saddle import aggregate_asr
    if args.asr_mode == "saddle":
        reps = []
        for r, t in zip(repos, teachers):
            if len(r) > 0:
                reps.append(r)  # SaddlePointRepository: lowest-loss snapshot is used
            else:  # fall back to final weights if no saddle qualified
                reps.append([[p.detach().clone().cpu() for p in t.parameters()]])
        return aggregate_asr(reps, select="best")
    if args.asr_mode == "last":
        return aggregate_asr([[[p.detach().clone().cpu() for p in t.parameters()]] for t in teachers])
    if args.asr_mode == "random_snapshot":
        g = torch.Generator().manual_seed(args.seed)
        picks = [[s[int(torch.randint(len(s), (1,), generator=g))]] for s in all_snaps]
        return aggregate_asr(picks)
    if args.asr_mode == "swa":
        avgs = []
        for s in all_snaps:
            avgs.append([[torch.stack([snap[i] for snap in s]).mean(0) for i in range(len(s[0]))]])
        return aggregate_asr(avgs)
    raise ValueError(args.asr_mode)


def main():
    args = parse()
    set_seed(args.seed)
    device = get_device(args.device)
    variant = "sprkd"
    if args.init_only: variant = "asrinit"
    parts = [variant, f"asr-{args.asr_mode}", f"T{args.teachers}x{args.teacher_epochs}ep"]
    if args.no_tm: parts.append("noTM")
    if args.no_nhe: parts.append("noNHE")
    if args.no_pgd: parts.append("noPGD")
    run_name = f"{args.model}_{'_'.join(parts)}_from_{args.teacher}" + ("" if args.dataset == "cifar100" else f"_{args.dataset}") + f"_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_name = args.run_name or run_name
    run_dir = Path(args.out_root) / run_name
    if args.resume and (run_dir / "summary.json").is_file():
        print(f"skip {run_name}: summary.json exists", flush=True); return
    logger = RunLogger(run_dir, {**vars(args), "run_name": run_name, "device": str(device), "mode": variant})

    train_loader, test_loader, num_classes = get_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers, args.seed, args.subset, download=True)
    args.num_classes = num_classes

    # ---- Phase 1: weak teachers (cached under <run_dir>/teachers.pt when --resume) ----
    from sprkd.saddle import SaddlePointRepository
    teachers, repos, all_snaps = [], [], []
    teacher_cache = Path(args.teacher_cache) if args.teacher_cache else run_dir / "teachers.pt"
    teacher_cache.parent.mkdir(parents=True, exist_ok=True)
    ck = safe_load(teacher_cache, map_location="cpu", weights_only=False) if (args.resume or args.teacher_cache) else None
    if ck is not None:
        for k, tk in enumerate(ck["teachers"]):
            t = build_model(args.teacher, num_classes).to(device); t.load_state_dict(tk["model"])
            repo = SaddlePointRepository(teacher_index=k)
            for snap, loss, gn, step, rule in zip(tk["snapshots"], tk["losses"], tk["grad_norms"], tk["steps"], tk["rules"]):
                repo.append([torch.nn.Parameter(x) for x in snap], loss=loss, grad_norm=gn, step=step, rule=rule)
            repo.n_checked = tk["n_checked"]
            teachers.append(t); repos.append(repo); all_snaps.append(tk["all_snaps"])
        logger.rows = ck.get("rows", [])
        print(f"resumed teacher phase from {teacher_cache}", flush=True)
    else:
        for k in range(args.teachers):
            t, repo, snaps = train_weak_teacher(args, k, train_loader, test_loader, device, logger)
            teachers.append(t); repos.append(repo); all_snaps.append(snaps)
            print(json.dumps({"event": "teacher_done", "teacher": k, "n_saddles": len(repo), "losses": [round(l, 4) for l in repo.losses][:20]}), flush=True)
        if args.resume or args.teacher_cache:
            atomic_save({"teachers": [{"model": t.state_dict(), "snapshots": r.snapshots, "losses": r.losses, "grad_norms": r.grad_norms,
                                      "steps": r.steps, "rules": r.rules, "n_checked": r.n_checked, "all_snaps": sn}
                                     for t, r, sn in zip(teachers, repos, all_snaps)], "rows": logger.rows}, teacher_cache)

    # ---- Phase 2: ASR + TLI ----
    # The package's inject_state_list() requires equal parameter counts, which fails for
    # depth-mismatched pairs. Load the ASR into a teacher-shaped module and use the
    # key-matching + center-crop primitive directly.
    import copy
    from sprkd.tli import simple_inject
    asr = build_asr(args, repos, all_snaps, teachers)
    asr_module = copy.deepcopy(teachers[0])
    with torch.no_grad():
        for tp, ts in zip(asr_module.parameters(), asr):
            tp.copy_(ts.to(tp.device, tp.dtype))
    student = build_model(args.model, num_classes).to(device)
    student_init = evaluate(student, test_loader, device)["top1"]
    injected_model = copy.deepcopy(student)
    pairs = simple_inject(injected_model, asr_module)
    injected = evaluate(injected_model, test_loader, device)
    targets = [p.detach().clone() for p in injected_model.parameters()]
    n_copied = sum(1 for s_k, t_k in pairs)
    logger.log_epoch(phase="inject", epoch=0, test_top1=injected["top1"], test_top5=injected["top5"], test_loss=injected["loss"],
                     random_init_top1=student_init, student_params=count_params(student), layers_paired=n_copied,
                     layers_total=len(list(student.parameters())))
    # Paper semantics: the student starts from its random init and the TM phase pulls it onto the
    # injected tensors. With --init-only or --no-tm the student starts directly at the injected tensors.
    if args.init_only or args.no_tm:
        student.load_state_dict(injected_model.state_dict())
    del injected_model, asr_module

    # ---- Phase 3: student ----
    base = make_sgd(student.parameters(), args.lr, weight_decay=args.wd)
    sched = make_multistep(base, args.milestones)
    if args.init_only:
        opt = base
    else:
        from sprkd.optimizer import SPRKD
        kw = dict(epsilon=args.epsilon, nhe_direction=args.nhe_direction)
        if args.no_nhe: kw["max_nhe_steps"] = 0
        if args.no_pgd: kw["pgd_perturb_variance"] = 0.0
        opt = SPRKD(student.parameters(), base_optimizer=base, loss_fn=nn.CrossEntropyLoss(),
                    teacher_saddle_points=[t.to(device) for t in targets], saddle_steps=None, **kw)
        if args.no_tm:
            opt._allow_targeting = {i: False for i, _ in enumerate(opt.param_groups[0]["params"])}
    probe = _probe_batch(train_loader, args.hessian_batch, device)
    student._steps_per_epoch = len(train_loader)
    best = -1.0
    tm_steps = None
    start_epoch = 1
    resume_path = run_dir / "resume.pt"
    ck = safe_load(resume_path, map_location=device, weights_only=False) if args.resume else None
    if ck is not None:
        student.load_state_dict(ck["model"]); base.load_state_dict(ck["base_opt"]); sched.load_state_dict(ck["sched"])
        best, start_epoch, tm_steps = ck["best"], ck["epoch"] + 1, ck.get("tm_steps")
        logger.rows = ck.get("rows", logger.rows)
        if not args.init_only:
            for k_, v_ in ck["sprkd_state"].items():
                setattr(opt, k_, v_)
        print(f"resumed student at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        loss_sum = n = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = student(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if args.init_only:
                opt.step()
            else:
                opt.step(model=student, current_loss=loss.detach(), data_batch=probe)
                student.train()
                if tm_steps is None and opt.at_asr():
                    tm_steps = opt.step_count
                    print(json.dumps({"event": "asr_reached", "step": tm_steps}), flush=True)
            loss_sum += loss.item() * y.numel(); n += y.numel(); correct += (logits.argmax(1) == y).sum().item()
        sched.step()
        ev = evaluate(student, test_loader, device)
        extra = {}
        if not args.init_only:
            c = opt.counters()
            extra = {"nhe_taken": c["nhe_taken"], "nhe_applied": c["nhe_applied"], "nhe_reverted": c["nhe_reverted"],
                     "pgd_fired": c["pgd_fired"], "pgd_reverted": c["pgd_reverted"], "tm_steps": tm_steps}
        logger.log_epoch(phase="student", epoch=epoch, lr=base.param_groups[0]["lr"], train_loss=loss_sum / n, train_top1=100 * correct / n,
                         test_top1=ev["top1"], test_top5=ev["top5"], test_loss=ev["loss"], **extra)
        if ev["top1"] > best:
            best = ev["top1"]
            if args.save_ckpt:
                atomic_save({"model": student.state_dict(), "epoch": epoch, "test_top1": best}, run_dir / "best.pt")
        if args.resume:
            sprkd_state = {} if args.init_only else {k_: getattr(opt, k_) for k_ in (
                "_step_count", "_allow_targeting", "_cooldown", "_stored_loss", "_tm_finished_step", "_n_nhe_taken", "_n_nhe_applied",
                "_n_nhe_no_negative", "_n_nhe_reverted", "_n_pgd_fired", "_n_pgd_reverted", "_n_pgd_considered", "_nhe_eigenvalues", "events")}
            atomic_save({"model": student.state_dict(), "base_opt": base.state_dict(), "sched": sched.state_dict(), "best": best, "epoch": epoch,
                        "tm_steps": tm_steps, "rows": logger.rows, "sprkd_state": sprkd_state}, resume_path)
    logger.finish(n_saddles=[len(r) for r in repos], saddle_grad_norms=[list(r.grad_norms) for r in repos],
                  saddle_losses=[list(r.losses) for r in repos], saddles_checked=[r.n_checked for r in repos],
                  injected_top1=injected["top1"], tm_steps=tm_steps,
                  counters=None if args.init_only else opt.counters())


if __name__ == "__main__":
    main()
