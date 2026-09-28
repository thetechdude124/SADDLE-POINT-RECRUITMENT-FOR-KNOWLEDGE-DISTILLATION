"""Saddle study on CIFAR-100 (non-saturated task), designed to run locally on CPU subsets for
smoke tests and on Modal (one container per stage/config) for the real thing.

Stages (each resumable, results under --out):
  teachers   : weak resnet32x4 teachers (CRD recipe, --teacher-epochs epochs) for the given
               seeds, checkpoints at --fractions of the run; per checkpoint: loss/acc, grad
               norm, lambda_max and the 5 most negative eigenvalues on a fixed probe batch.
  refine     : refine every checkpoint to a stationary point (--refine-method, fixed
               tolerance) on the probe batch; record grad norm before/after, loss, val
               accuracy, lambda extremes, distance moved; classify by convergence.
  ridge      : string method + climbing image between two teacher minima (pairs), refined
               and verified; path losses on the full train set and the test set.
  transplant : inject a chosen point (ridge / raw checkpoint / refined checkpoint / final
               weights) into resnet8x4 and train with the 240-epoch CRD schedule (or
               --student-epochs); reports final and best test top-1 and epochs to a
               threshold. Scratch and KD-weak are the reference arms from train.py.

Examples
  python cifar_saddle_study.py --stage teachers --seeds 0 --teacher-epochs 10 --fractions 0.1,0.5,1.0
  python cifar_saddle_study.py --stage refine   --seeds 0
  python cifar_saddle_study.py --stage ridge    --pairs 0:1
  python cifar_saddle_study.py --stage transplant --point ridge:0:1 --student-seeds 0,1,2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

from common import RunLogger, evaluate, get_device, make_multistep, make_sgd, set_seed
from data import get_loaders
from models import build_model

HERE = Path(__file__).resolve().parent


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["teachers", "refine", "ridge", "transplant"])
    p.add_argument("--out", default=str(HERE / "results" / "cifar_saddle"))
    p.add_argument("--data-root", default=str(HERE / "data"))
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--teacher", default="resnet32x4")
    p.add_argument("--student", default="resnet8x4")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--teacher-epochs", type=int, default=10)
    p.add_argument("--fractions", default="0.02,0.05,0.1,0.2,0.35,0.5,0.65,0.8,1.0")
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--refine-method", default="adam", choices=["adam", "lm", "gn", "gradnorm", "hisd"])
    p.add_argument("--hisd-index", type=int, default=1)
    p.add_argument("--refine-steps", type=int, default=2000)
    p.add_argument("--refine-lr", type=float, default=1e-3)
    p.add_argument("--refine-grad-tol", type=float, default=1e-2)
    p.add_argument("--pairs", default="0:1")
    p.add_argument("--nodes", type=int, default=10)
    p.add_argument("--neb-images", type=int, default=2048)
    p.add_argument("--string-iters", type=int, default=200)
    p.add_argument("--climb-iters", type=int, default=200)
    p.add_argument("--neb-lr", type=float, default=0.01)
    p.add_argument("--max-step", type=float, default=0.5)
    p.add_argument("--point", default=None, help="transplant source: ridge:A:B | ck:SEED:FRAC | ref:SEED:FRAC | hisd:SEED:FRAC | final:SEED")
    p.add_argument("--student-seeds", default="0,1,2")
    p.add_argument("--student-epochs", type=int, default=240)
    p.add_argument("--milestones", type=int, nargs="*", default=[150, 180, 210])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--threshold", type=float, default=60.0, help="test top-1 threshold for epochs-to-threshold")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--subset", type=int, default=None, help="train subset for smoke tests")
    p.add_argument("--device", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- helpers

def flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat(model, vec):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel(); p.copy_(vec[i:i + n].view_as(p).to(p.device)); i += n


def probe_batch(train_loader, n, seed=12345):
    xs, ys = [], []
    g = torch.Generator().manual_seed(seed)
    for x, y in train_loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= 4 * n:
            break
    x = torch.cat(xs); y = torch.cat(ys)
    idx = torch.randperm(len(y), generator=g)[:n]
    return x[idx], y[idx]


def curvature(model, batch, k=5):
    from sprkd.hessian_utils import extreme_eigenpairs
    m = copy.deepcopy(model).cpu()
    e = extreme_eigenpairs(m, nn.CrossEntropyLoss(), batch, k=k, tol=1e-2, max_iter=20)
    return {"lambda_max": e["lambda_max"], "lambda_min_k": e["lambda_min_k"], "grad_norm_probe": e["grad_norm"], "n_hvp": e["n_hvp"]}


def train_teacher_with_ckpts(args, seed, train_loader, test_loader, device, out):
    d = Path(out) / f"teacher_s{seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "checkpoints.json").is_file():
        return json.load(open(d / "checkpoints.json"))
    set_seed(seed)
    t = build_model(args.teacher, 100).to(device)
    opt = make_sgd(t.parameters(), args.lr)
    total = args.teacher_epochs * len(train_loader)
    fr = [float(f) for f in args.fractions.split(",")]
    ck_steps = {max(1, round(f * total)): f for f in fr}
    step = 0; recs = []; t0 = time.time()
    for ep in range(args.teacher_epochs):
        t.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True); loss = F.cross_entropy(t(x), y); loss.backward(); opt.step(); step += 1
            if step in ck_steps:
                torch.save(t.state_dict(), d / f"ck_{int(round(ck_steps[step] * 100)):03d}.pt")
                ev = evaluate(t, test_loader, device)
                recs.append({"fraction": ck_steps[step], "step": step, "test_top1": ev["top1"], "test_loss": ev["loss"], "elapsed_s": round(time.time() - t0, 1)})
                print(json.dumps({"seed": seed, **recs[-1]}), flush=True)
    json.dump({"seed": seed, "steps_total": total, "records": recs}, open(d / "checkpoints.json", "w"), indent=2)
    return {"seed": seed, "steps_total": total, "records": recs}


# --------------------------------------------------------------------------- stages

def stage_teachers(args, device):
    train_loader, test_loader, _ = get_loaders(args.dataset, args.data_root, 64, args.num_workers, 0, args.subset, download=True)
    probe = probe_batch(train_loader, args.probe_size)
    torch.save(probe, Path(args.out) / "probe.pt")
    for seed in [int(s) for s in args.seeds.split(",")]:
        ck = train_teacher_with_ckpts(args, seed, train_loader, test_loader, device, args.out)
        d = Path(args.out) / f"teacher_s{seed}"
        if (d / "metrics.json").is_file():
            continue
        rows = []
        for r in ck["records"]:
            t = build_model(args.teacher, 100).to(device); t.load_state_dict(torch.load(d / f"ck_{int(round(r['fraction'] * 100)):03d}.pt", map_location=device))
            t0 = time.time(); c = curvature(t, probe)
            rows.append({**r, **c, "curv_wall_s": round(time.time() - t0, 1)})
            print(json.dumps({"seed": seed, "fraction": r["fraction"], "lmax": round(c["lambda_max"], 2), "lmin_k": [round(v, 3) for v in c["lambda_min_k"]], "gn": round(c["grad_norm_probe"], 4), "s": rows[-1]["curv_wall_s"]}), flush=True)
            json.dump(rows, open(d / "metrics_partial.json", "w"), indent=2)
        json.dump(rows, open(d / "metrics.json", "w"), indent=2)


def stage_refine(args, device):
    from sprkd.saddle import refine_to_stationary
    _, test_loader, _ = get_loaders(args.dataset, args.data_root, 64, args.num_workers, 0, args.subset, download=True)
    probe = torch.load(Path(args.out) / "probe.pt")
    for seed in [int(s) for s in args.seeds.split(",")]:
        d = Path(args.out) / f"teacher_s{seed}"
        tag = "hisd" if args.refine_method == "hisd" else "ref"
        out_json = d / ("hisd.json" if tag == "hisd" else "refined.json")
        done = json.load(open(out_json)) if out_json.is_file() else {}
        for f in [float(x) for x in args.fractions.split(",")]:
            key = f"{int(round(f * 100)):03d}"
            if key in done:
                continue
            t = build_model(args.teacher, 100); t.load_state_dict(torch.load(d / f"ck_{key}.pt", map_location="cpu"))
            th0 = flat(t); t0 = time.time()
            rec = refine_to_stationary(t, nn.CrossEntropyLoss(), probe, max_steps=args.refine_steps, grad_tol=args.refine_grad_tol, lr=args.refine_lr, method=args.refine_method, solver_iters=50, hisd_index=args.hisd_index)
            c = curvature(t, probe)
            ev = evaluate(copy.deepcopy(t).to(device), test_loader, device)
            cls = ("converged_saddle" if c["lambda_min_k"][0] < -0.1 else "converged_minimum_or_flat") if rec["converged"] else ("unconverged_negcurv" if c["lambda_min_k"][0] < 0 else "unconverged")
            done[key] = {"fraction": f, "refine": rec, **c, "test": ev, "dist_rel": float((flat(t) - th0).norm() / th0.norm()), "class": cls, "wall_s": round(time.time() - t0, 1)}
            torch.save(t.state_dict(), d / f"{tag}_{key}.pt")
            json.dump(done, open(out_json, "w"), indent=2)
            print(json.dumps({"seed": seed, "fraction": f, "gn": (round(rec["grad_norm_before"], 4), round(rec["grad_norm_after"], 5)), "loss": (round(rec["loss_before"], 4), round(rec["loss_after"], 4)), "test_top1": round(ev["top1"], 2), "lmin": round(c["lambda_min_k"][0], 4), "class": cls, "s": done[key]["wall_s"]}), flush=True)


class FlatLoss:
    def __init__(self, model, x, y):
        self.model = copy.deepcopy(model).eval()
        self.names = [n for n, _ in self.model.named_parameters()]
        self.shapes = [p.shape for _, p in self.model.named_parameters()]
        self.numels = [p.numel() for _, p in self.model.named_parameters()]
        self.x, self.y = x, y

    def unflat(self, v):
        out, i = {}, 0
        for n, shp, k in zip(self.names, self.shapes, self.numels):
            out[n] = v[i:i + k].view(shp); i += k
        return out

    def loss_and_grad(self, v):
        v = v.detach().requires_grad_(True)
        l = F.cross_entropy(functional_call(self.model, self.unflat(v), (self.x,)), self.y)
        g, = torch.autograd.grad(l, v)
        return float(l), g.detach()

    @torch.no_grad()
    def loss_loader(self, v, loader, device):
        tot = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            tot += float(F.cross_entropy(functional_call(self.model, self.unflat(v), (x,)), y, reduction="sum")); n += len(y)
        return tot / n


def redistribute(nodes):
    seg = torch.stack([(nodes[i + 1] - nodes[i]).norm() for i in range(len(nodes) - 1)])
    cum = torch.cat([torch.zeros(1, device=seg.device), torch.cumsum(seg, 0)])
    targets = torch.linspace(0, float(cum[-1]), len(nodes), device=seg.device)
    new = [nodes[0]]
    for t in targets[1:-1]:
        j = max(0, min(int(torch.searchsorted(cum, t, right=True).item()) - 1, len(nodes) - 2))
        a = (t - cum[j]) / max(float(seg[j]), 1e-12)
        new.append(nodes[j] + a * (nodes[j + 1] - nodes[j]))
    return new + [nodes[-1]]


def tangent(nodes, losses, i):
    lp, l0, ln = losses[i - 1], losses[i], losses[i + 1]
    tp, tn = nodes[i] - nodes[i - 1], nodes[i + 1] - nodes[i]
    if ln > l0 > lp: tau = tn
    elif ln < l0 < lp: tau = tp
    else:
        dmax, dmin = max(abs(ln - l0), abs(lp - l0)), min(abs(ln - l0), abs(lp - l0))
        tau = tn * dmax + tp * dmin if ln > lp else tn * dmin + tp * dmax
    return tau / (tau.norm() + 1e-12)


def stage_ridge(args, device):
    from sprkd.saddle import refine_to_stationary
    train_loader, test_loader, _ = get_loaders(args.dataset, args.data_root, 64, args.num_workers, 0, args.subset, download=True)
    probe = torch.load(Path(args.out) / "probe.pt")
    xs, ys = [], []
    for x, y in train_loader:
        xs.append(x); ys.append(y)
        if sum(len(t) for t in ys) >= args.neb_images: break
    x_neb, y_neb = torch.cat(xs)[:args.neb_images].to(device), torch.cat(ys)[:args.neb_images].to(device)
    out_f = Path(args.out) / "ridge.json"
    done = json.load(open(out_f)) if out_f.is_file() else {}
    for pair in args.pairs.split(","):
        sa, sb = [int(v) for v in pair.split(":")]; key = f"pair_{sa}_{sb}"
        if key in done: continue
        t0 = time.time()
        tA = build_model(args.teacher, 100).to(device); tA.load_state_dict(torch.load(Path(args.out) / f"teacher_s{sa}" / "ck_100.pt", map_location=device))
        tB = build_model(args.teacher, 100).to(device); tB.load_state_dict(torch.load(Path(args.out) / f"teacher_s{sb}" / "ck_100.pt", map_location=device))
        fl = FlatLoss(tA, x_neb, y_neb); a, b = flat(tA), flat(tB)
        K = args.nodes; nodes = [a + (b - a) * (i / (K + 1)) for i in range(K + 2)]
        lin = [fl.loss_loader(a + float(t) * (b - a), test_loader, device) for t in torch.linspace(0, 1, 6)]
        hist = []
        for it in range(args.string_iters):
            lg = [fl.loss_and_grad(v) for v in nodes]; losses = [l for l, _ in lg]
            if any(math.isnan(l) for l in losses): raise RuntimeError("string diverged")
            for i in range(1, K + 1):
                tau = tangent(nodes, losses, i); g = lg[i][1]
                step = args.neb_lr * (g - torch.dot(g, tau) * tau); sn = float(step.norm())
                nodes[i] = nodes[i] - (step * (args.max_step / sn) if sn > args.max_step else step)
            if it % 10 == 9: nodes = redistribute(nodes)
            if it % 25 == 0: hist.append({"phase": "string", "iter": it, "losses": [round(l, 4) for l in losses]}); print(json.dumps({key: hist[-1]}), flush=True)
        losses = [fl.loss_and_grad(v)[0] for v in nodes]; ci = max(range(1, K + 1), key=lambda j: losses[j])
        for it in range(args.climb_iters):
            losses = [fl.loss_and_grad(v)[0] for v in nodes]; l, g = fl.loss_and_grad(nodes[ci]); tau = tangent(nodes, losses, ci)
            step = args.neb_lr * (g - 2.0 * torch.dot(g, tau) * tau); sn = float(step.norm())
            nodes[ci] = nodes[ci] - (step * (args.max_step / sn) if sn > args.max_step else step)
            if it % 50 == 0 or it == args.climb_iters - 1: hist.append({"phase": "climb", "iter": it, "loss": l, "grad_norm": float(g.norm())}); print(json.dumps({key: hist[-1]}), flush=True)
        path_test = [fl.loss_loader(v, test_loader, device) for v in nodes]
        m_ci = build_model(args.teacher, 100).to(device); load_flat(m_ci, nodes[ci])
        torch.save(m_ci.state_dict(), Path(args.out) / f"{key}_climb.pt")
        m_ref = copy.deepcopy(m_ci).cpu(); t1 = time.time()
        rf = refine_to_stationary(m_ref, nn.CrossEntropyLoss(), probe, max_steps=args.refine_steps, grad_tol=args.refine_grad_tol, lr=args.refine_lr, method=args.refine_method, solver_iters=50)
        c = curvature(m_ref, probe); ev = evaluate(copy.deepcopy(m_ref).to(device), test_loader, device)
        torch.save(m_ref.state_dict(), Path(args.out) / f"{key}_refined.pt")
        done[key] = {"pair": [sa, sb], "endpoint_test": [evaluate(tA, test_loader, device), evaluate(tB, test_loader, device)], "endpoint_l2": float((a - b).norm()),
                     "linear_path_test_loss": lin, "string_path_subset_loss": losses, "string_path_test_loss": path_test, "climbing_index": ci,
                     "barrier_test": max(path_test) - 0.5 * (path_test[0] + path_test[-1]), "climb_test": evaluate(m_ci, test_loader, device),
                     "refined": {"refine": rf, **c, "test": ev, "refine_wall_s": round(time.time() - t1, 1)}, "history": hist, "wall_s": round(time.time() - t0, 1)}
        json.dump(done, open(out_f, "w"), indent=2)
        print(json.dumps({key: {k: v for k, v in done[key].items() if k not in ("history", "string_path_subset_loss")}}), flush=True)


def stage_transplant(args, device):
    from sprkd.tli import simple_inject
    assert args.point, "--point required"
    kind, *rest = args.point.split(":")
    out = Path(args.out)
    if kind == "ridge":
        src_state = torch.load(out / f"pair_{rest[0]}_{rest[1]}_refined.pt", map_location=device); name = f"ridge_{rest[0]}_{rest[1]}"
    elif kind == "climb":
        src_state = torch.load(out / f"pair_{rest[0]}_{rest[1]}_climb.pt", map_location=device); name = f"climb_{rest[0]}_{rest[1]}"
    elif kind in ("ck", "ref", "hisd"):
        src_state = torch.load(out / f"teacher_s{rest[0]}" / f"{kind}_{int(round(float(rest[1]) * 100)):03d}.pt", map_location=device); name = f"{kind}_s{rest[0]}_f{int(round(float(rest[1]) * 100)):03d}"
    elif kind == "final":
        src_state = torch.load(out / f"teacher_s{rest[0]}" / "ck_100.pt", map_location=device); name = f"final_s{rest[0]}"
    else:
        raise ValueError(args.point)
    train_loader, test_loader, _ = get_loaders(args.dataset, args.data_root, 64, args.num_workers, 0, args.subset, download=True)
    for ss in [int(s) for s in args.student_seeds.split(",")]:
        run_dir = out / "transplant" / f"{args.student}_{name}_s{ss}"
        if (run_dir / "summary.json").is_file():
            print(f"skip {run_dir.name}"); continue
        set_seed(ss)
        student = build_model(args.student, 100).to(device)
        src = build_model(args.teacher, 100).to(device); src.load_state_dict(src_state); simple_inject(student, src)
        logger = RunLogger(run_dir, {**vars(args), "mode": "transplant", "model": args.student, "teacher": args.teacher, "point": args.point, "arm": name, "seed": ss, "epochs": args.student_epochs, "dataset": args.dataset})
        init = evaluate(student, test_loader, device)
        opt = make_sgd(student.parameters(), args.lr); sched = make_multistep(opt, args.milestones)
        best = -1.0; ep_thr = None
        for epoch in range(1, args.student_epochs + 1):
            student.train(); ls = n = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True); loss = F.cross_entropy(student(x), y); loss.backward(); opt.step()
                ls += loss.item() * len(y); n += len(y)
            sched.step(); ev = evaluate(student, test_loader, device)
            if ep_thr is None and ev["top1"] >= args.threshold: ep_thr = epoch
            best = max(best, ev["top1"])
            logger.log_epoch(epoch=epoch, train_loss=ls / n, test_top1=ev["top1"], test_top5=ev["top5"], test_loss=ev["loss"])
        logger.finish(init_test_top1=init["top1"], epochs_to_threshold=ep_thr, threshold=args.threshold)


def main():
    args = parse()
    device = get_device(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"teachers": stage_teachers, "refine": stage_refine, "ridge": stage_ridge, "transplant": stage_transplant}[args.stage](args, device)


if __name__ == "__main__":
    main()
