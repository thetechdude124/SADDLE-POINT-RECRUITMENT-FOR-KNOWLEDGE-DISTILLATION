"""Saddle timing study on malaria (see neurips/07_saddle_timing_analysis.md).

Stages (all resumable; results under results/saddle_timing/):
  1 checkpoints : one normalised-input teacher per seed, 40 epochs, checkpoints at fixed
                  fractions of training; per checkpoint: loss/acc/grad norm/lambda_max/5 most
                  negative eigenvalues/index estimate/trace on a fixed 256-image probe batch
                  and loss/acc/grad norm/eigen-extremes on the full validation set.
  2 refine      : refine every checkpoint to a stationary point (fixed tolerance) and
                  classify the result (trivial saddle / informative saddle / minimum).
  3 basins      : SGD continuations (4 data-order seeds) from raw and refined points at
                  selected fractions; disagreement, linear-path loss barrier, L2 distance.
  4 transplant  : inject raw and refined points into the student, 30 epochs x 3 seeds;
                  test accuracy at best val and epochs to 90% val; scratch baseline.

    python saddle_timing_study.py --stage all
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import get_device, set_seed
from malaria_e1 import load_cached, make_split

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "saddle_timing"
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
BASIN_FRACTIONS = [0.01, 0.20, 0.35, 0.50, 0.65, 1.00]
TAU_ABS = 0.1
LN2 = math.log(2.0)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all", help="all | checkpoints | refine | basins | transplant")
    p.add_argument("--teacher-seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--student-epochs", type=int, default=30)
    p.add_argument("--student-seeds", default="0,1,2")
    p.add_argument("--transplant-teacher-seeds", default="0")
    p.add_argument("--basin-cont-seeds", type=int, default=4)
    p.add_argument("--control-100-epochs", type=int, default=10, help="continuation length from the 100% checkpoint")
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--refine-grad-tol", type=float, default=1e-2)
    p.add_argument("--refine-max-steps", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--fractions", default=None, help="override FRACTIONS, comma-separated")
    return p.parse_args()


# --------------------------------------------------------------------------- data

class Data:
    def __init__(self, device, probe_size):
        x, y, _ = load_cached(str(HERE.parents[1] / "Saddle-point-recruitment-for-knowledge-distillation" / "cell_images"), str(HERE / "data" / "malaria_32.pt"))
        split = make_split(y, str(HERE / "results" / "malaria_split.json"))
        tr = torch.tensor(split["train"]); va = torch.tensor(split["val"]); te = torch.tensor(split["test"])
        xf = x.float() / 255.0
        mean = xf[tr].mean(dim=(0, 2, 3)); std = xf[tr].std(dim=(0, 2, 3))
        self.mean, self.std = mean.tolist(), std.tolist()
        norm = (xf - mean[None, :, None, None]) / std[None, :, None, None]
        self.device = device
        self.x_tr, self.y_tr = norm[tr].to(device), y[tr].to(device)
        self.x_va, self.y_va = norm[va].to(device), y[va].to(device)
        self.x_te, self.y_te = norm[te].to(device), y[te].to(device)
        g = torch.Generator().manual_seed(12345)
        pi = torch.randperm(len(tr), generator=g)[:probe_size]
        self.probe = (self.x_tr[pi].cpu(), self.y_tr[pi].cpu())     # CPU copy for Hessian work
        self.val_cpu = (self.x_va.cpu(), self.y_va.cpu())
        gv = torch.Generator().manual_seed(54321)
        vi = torch.randperm(len(va), generator=gv)[:512]
        self.val_hess = (self.x_va[vi].cpu(), self.y_va[vi].cpu())   # fixed 512-image val subset for Hessian work

    def train_batches(self, seed, batch_size=64):
        g = torch.Generator().manual_seed(seed)
        n = len(self.y_tr)
        while True:
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                yield self.x_tr[idx], self.y_tr[idx]

    @property
    def steps_per_epoch(self):
        return math.ceil(len(self.y_tr) / 64)


@torch.no_grad()
def evaluate(model, x, y, bs=1024):
    model.eval()
    loss = 0.0; correct = 0; preds = []
    for i in range(0, len(y), bs):
        o = model(x[i:i + bs]); loss += F.cross_entropy(o, y[i:i + bs], reduction="sum").item()
        p = o.argmax(1); correct += (p == y[i:i + bs]).sum().item(); preds.append(p)
    return {"loss": loss / len(y), "acc": 100.0 * correct / len(y)}, torch.cat(preds)


def grad_norm_on(model, x, y):
    model.eval()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    g = torch.autograd.grad(loss, [p for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return float(torch.sqrt(sum((gi ** 2).sum() for gi in g)))


def flat(model):
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def load_flat(model, vec):
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel(); p.copy_(vec[i:i + n].view_as(p).to(p.device)); i += n


def hessian_stats(model, batch, k=5, do_density=True, eig_tol=1e-2, eig_max_iter=20):
    """lambda_max, k most negative eigenvalues, count < -tau, trace (Hutchinson), index estimate (SLQ)."""
    from sprkd.hessian_utils import HessianOperator, extreme_eigenpairs, hessian_compatible
    m = copy.deepcopy(model).cpu()
    t_e = time.time()
    ext = extreme_eigenpairs(m, nn.CrossEntropyLoss(), batch, k=k, tol=eig_tol, max_iter=eig_max_iter)
    out = {"lambda_max": ext["lambda_max"], "lambda_min_k": ext["lambda_min_k"], "eig_wall_s": round(time.time() - t_e, 1),
           "n_below_tau": sum(1 for v in ext["lambda_min_k"] if v < -TAU_ABS), "n_hvp_eigs": ext["n_hvp"], "grad_norm_probe": ext["grad_norm"]}
    with hessian_compatible(m, batch) as (mm, b, _):
        mm.eval()
        op = HessianOperator(mm, nn.CrossEntropyLoss(), b)
        tr = []
        for _ in range(20):
            v = torch.randint(0, 2, (op.n,)).float() * 2 - 1
            tr.append(float(torch.dot(v, op.hvp(v))))
        out["trace_hutchinson"] = sum(tr) / len(tr)
        out["n_params"] = op.n
        if do_density:
            # Lanczos quadrature (SLQ) with m=60 steps, 4 probes: fraction of spectral mass below -tau and below 0
            neg_mass, negtau_mass = [], []
            for _ in range(4):
                mlan = 60
                v = torch.randn(op.n); v = v / v.norm()
                alphas, betas = [], []; vprev = torch.zeros(op.n); beta = 0.0
                for j in range(mlan):
                    w = op.hvp(v) - beta * vprev
                    alpha = float(torch.dot(w, v)); w = w - alpha * v
                    beta = float(w.norm()); alphas.append(alpha); betas.append(beta)
                    if beta < 1e-8: break
                    vprev, v = v, w / beta
                T = torch.diag(torch.tensor(alphas)) + torch.diag(torch.tensor(betas[:-1]), 1) + torch.diag(torch.tensor(betas[:-1]), -1)
                evals, evecs = torch.linalg.eigh(T)
                weights = evecs[0] ** 2
                neg_mass.append(float(weights[evals < 0].sum())); negtau_mass.append(float(weights[evals < -TAU_ABS].sum()))
            out["index_estimate"] = sum(neg_mass) / len(neg_mass) * op.n
            out["index_estimate_below_tau"] = sum(negtau_mass) / len(negtau_mass) * op.n
            out["neg_spectral_mass"] = sum(neg_mass) / len(neg_mass)
        op.release()
    return out


# --------------------------------------------------------------------------- stage 1

def stage_checkpoints(args, data, device):
    from sprkd.models import MalariaTeacherCNN
    steps_total = args.epochs * data.steps_per_epoch
    ck_steps = {f: max(1, round(f * steps_total)) for f in FRACTIONS}
    for seed in [int(s) for s in args.teacher_seeds.split(",")]:
        d = OUT / f"teacher_s{seed}"; d.mkdir(parents=True, exist_ok=True)
        if (d / "checkpoints.json").is_file():
            print(f"stage1 seed {seed}: done", flush=True); continue
        set_seed(seed)
        t = MalariaTeacherCNN().to(device); opt = torch.optim.Adam(t.parameters(), 1e-3)
        batches = data.train_batches(seed); recs = []; t0 = time.time()
        for step in range(1, steps_total + 1):
            t.train(); x, y = next(batches)
            opt.zero_grad(); loss = F.cross_entropy(t(x), y); loss.backward(); opt.step()
            for f, s_ in ck_steps.items():
                if step == s_:
                    torch.save(t.state_dict(), d / f"ck_{int(f * 100):03d}.pt")
                    va, _ = evaluate(t, data.x_va, data.y_va)
                    recs.append({"fraction": f, "step": step, "epoch": step / data.steps_per_epoch, "train_batch_loss": float(loss), "val": va, "elapsed_s": round(time.time() - t0, 1)})
                    print(json.dumps({"seed": seed, **recs[-1]}), flush=True)
        json.dump({"seed": seed, "steps_total": steps_total, "steps_per_epoch": data.steps_per_epoch, "ck_steps": ck_steps, "records": recs,
                   "train_s": round(time.time() - t0, 1), "norm_mean": data.mean, "norm_std": data.std}, open(d / "checkpoints.json", "w"), indent=2)
    # metrics per checkpoint
    for seed in [int(s) for s in args.teacher_seeds.split(",")]:
        d = OUT / f"teacher_s{seed}"
        if (d / "metrics.json").is_file():
            continue
        part = d / "metrics_partial.json"
        rows = json.load(open(part)) if part.is_file() else []
        done_f = {r["fraction"] for r in rows}
        for f in FRACTIONS:
            if f in done_f:
                continue
            t = MalariaTeacherCNN().to(device); t.load_state_dict(torch.load(d / f"ck_{int(f * 100):03d}.pt", map_location=device))
            t0 = time.time()
            pr = {"loss_probe": F.cross_entropy(t.eval()(data.probe[0].to(device)), data.probe[1].to(device)).item(),
                  "acc_probe": float((t(data.probe[0].to(device)).argmax(1) == data.probe[1].to(device)).float().mean() * 100)}
            hp = hessian_stats(t, data.probe, k=5, do_density=True)
            va, _ = evaluate(t, data.x_va, data.y_va)
            gn_val = grad_norm_on(copy.deepcopy(t).cpu(), *data.val_cpu)
            hv = hessian_stats(t, data.val_hess, k=1, do_density=False)   # 512-image subset; full-val Hessian took 17-36 min per checkpoint
            rows.append({"fraction": f, **pr, "probe": hp, "val": {**va, "grad_norm": gn_val, **hv}, "wall_s": round(time.time() - t0, 1)})
            json.dump(rows, open(part, "w"), indent=2)
            print(json.dumps({"seed": seed, "fraction": f, "probe_lmax": round(hp["lambda_max"], 3), "probe_lmin_k": [round(v, 3) for v in hp["lambda_min_k"]], "probe_gn": round(hp["grad_norm_probe"], 4),
                              "index_est": round(hp.get("index_estimate", float("nan")), 1), "val_acc": round(va["acc"], 2), "val_lmin": round(hv["lambda_min_k"][0], 3), "wall_s": rows[-1]["wall_s"]}), flush=True)
        json.dump(rows, open(d / "metrics.json", "w"), indent=2)


# --------------------------------------------------------------------------- stage 2

def classify(refined_loss, val_acc, lambda_min):
    if lambda_min >= 0:
        return "minimum"
    if abs(refined_loss - LN2) <= 0.02 * LN2 or val_acc < 55.0:
        return "trivial_saddle"
    if lambda_min < -TAU_ABS and val_acc > 70.0:
        return "informative_saddle"
    return "weak_saddle"   # lambda_min in (-tau, 0) or 55 <= acc <= 70


def stage_refine(args, data, device):
    from sprkd.models import MalariaTeacherCNN
    from sprkd.hessian_utils import extreme_eigenpairs
    from sprkd.saddle import refine_to_stationary
    for seed in [int(s) for s in args.teacher_seeds.split(",")]:
        d = OUT / f"teacher_s{seed}"
        if (d / "refined.json").is_file():
            continue
        rows = []
        for f in FRACTIONS:
            t = MalariaTeacherCNN(); t.load_state_dict(torch.load(d / f"ck_{int(f * 100):03d}.pt", map_location="cpu"))
            theta0 = flat(t)
            t0 = time.time()
            rec = refine_to_stationary(t, nn.CrossEntropyLoss(), data.probe, max_steps=args.refine_max_steps, grad_tol=args.refine_grad_tol)
            ext = extreme_eigenpairs(t, nn.CrossEntropyLoss(), data.probe, k=5, tol=1e-2, max_iter=20)
            theta1 = flat(t)
            torch.save(t.state_dict(), d / f"ref_{int(f * 100):03d}.pt")
            tt = copy.deepcopy(t).to(device)
            va, _ = evaluate(tt, data.x_va, data.y_va)
            va_gn = grad_norm_on(copy.deepcopy(t).cpu(), *data.val_cpu)
            dist = float((theta1 - theta0).norm())
            row = {"fraction": f, "refine": rec, "refined_loss_probe": rec["loss_after"], "refined_val": va, "refined_val_grad_norm": va_gn,
                   "lambda_max_after": ext["lambda_max"], "lambda_min_after": ext["lambda_min"], "lambda_min_k_after": ext["lambda_min_k"],
                   "dist_l2": dist, "dist_rel": dist / float(theta0.norm()), "theta0_norm": float(theta0.norm()),
                   "class": classify(rec["loss_after"], va["acc"], ext["lambda_min"]), "wall_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(json.dumps({"seed": seed, "fraction": f, "gn": (round(rec["grad_norm_before"], 4), round(rec["grad_norm_after"], 5)), "loss": (round(rec["loss_before"], 4), round(rec["loss_after"], 4)),
                              "val_acc": round(va["acc"], 2), "lmin_after": round(ext["lambda_min"], 4), "dist": round(dist, 3), "rel": round(row["dist_rel"], 4), "steps": rec["steps"], "class": row["class"], "s": row["wall_s"]}), flush=True)
        json.dump(rows, open(d / "refined.json", "w"), indent=2)


# --------------------------------------------------------------------------- stage 3

def continue_from(state, data, device, n_epochs, seed):
    from sprkd.models import MalariaTeacherCNN
    t = MalariaTeacherCNN().to(device); t.load_state_dict(state)
    set_seed(1000 + seed)
    opt = torch.optim.Adam(t.parameters(), 1e-3); batches = data.train_batches(5000 + seed)
    for _ in range(n_epochs * data.steps_per_epoch):
        t.train(); x, y = next(batches); opt.zero_grad(); F.cross_entropy(t(x), y).backward(); opt.step()
    return t


@torch.no_grad()
def path_barrier(model, ta, tb, data, n=11):
    losses = []
    for a in torch.linspace(0, 1, n):
        load_flat(model, ta + a * (tb - ta))
        losses.append(evaluate(model, data.x_va, data.y_va)[0]["loss"])
    return {"losses": losses, "barrier": max(losses) - 0.5 * (losses[0] + losses[-1])}


def stage_basins(args, data, device):
    from sprkd.models import MalariaTeacherCNN
    for seed in [int(s) for s in args.teacher_seeds.split(",")]:
        d = OUT / f"teacher_s{seed}"; out_f = d / "basins.json"
        done = json.load(open(out_f)) if out_f.is_file() else {}
        ck = json.load(open(d / "checkpoints.json"))
        for f, kind in itertools.product(BASIN_FRACTIONS, ("raw", "refined")):
            key = f"{kind}_{int(f * 100):03d}"
            if key in done:
                continue
            t0 = time.time()
            state = torch.load(d / (f"ck_{int(f * 100):03d}.pt" if kind == "raw" else f"ref_{int(f * 100):03d}.pt"), map_location=device)
            remaining = args.epochs - int(round(f * args.epochs)) if f < 1.0 else args.control_100_epochs
            ends, preds, accs = [], [], []
            for s_ in range(args.basin_cont_seeds):
                m = continue_from(state, data, device, remaining, s_)
                ev, p = evaluate(m, data.x_va, data.y_va)
                ends.append(flat(m)); preds.append(p.cpu()); accs.append(ev["acc"])
            probe_model = MalariaTeacherCNN().to(device)
            pairs = []
            for i, j in itertools.combinations(range(len(ends)), 2):
                pb = path_barrier(probe_model, ends[i], ends[j], data)
                pairs.append({"pair": [i, j], "disagreement": float((preds[i] != preds[j]).float().mean()), "l2": float((ends[i] - ends[j]).norm()), **pb})
            done[key] = {"fraction": f, "kind": kind, "remaining_epochs": remaining, "end_val_acc": accs,
                         "mean_disagreement": sum(p["disagreement"] for p in pairs) / len(pairs), "mean_barrier": sum(p["barrier"] for p in pairs) / len(pairs),
                         "max_barrier": max(p["barrier"] for p in pairs), "mean_l2": sum(p["l2"] for p in pairs) / len(pairs), "pairs": pairs, "wall_s": round(time.time() - t0, 1)}
            json.dump(done, open(out_f, "w"), indent=2)
            print(json.dumps({"seed": seed, "key": key, "end_acc": [round(a, 2) for a in accs], "disagree": round(done[key]["mean_disagreement"], 4), "barrier": round(done[key]["mean_barrier"], 4), "l2": round(done[key]["mean_l2"], 3), "s": done[key]["wall_s"]}), flush=True)


# --------------------------------------------------------------------------- stage 4

def train_student(init_state_teacher, data, device, seed, n_epochs):
    from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
    from sprkd.tli import simple_inject
    set_seed(seed)
    s = MalariaStudentCNN().to(device)
    if init_state_teacher is not None:
        src = MalariaTeacherCNN().to(device); src.load_state_dict(init_state_teacher); simple_inject(s, src)
    init_val, _ = evaluate(s, data.x_va, data.y_va)
    opt = torch.optim.Adam(s.parameters(), 1e-3); batches = data.train_batches(7000 + seed)
    best_val, best_test, ep90, hist = -1.0, None, None, []
    for ep in range(1, n_epochs + 1):
        for _ in range(data.steps_per_epoch):
            s.train(); x, y = next(batches); opt.zero_grad(); F.cross_entropy(s(x), y).backward(); opt.step()
        va, _ = evaluate(s, data.x_va, data.y_va); te, _ = evaluate(s, data.x_te, data.y_te)
        hist.append({"epoch": ep, "val_acc": va["acc"], "test_acc": te["acc"]})
        if va["acc"] > best_val:
            best_val, best_test = va["acc"], te["acc"]
        if ep90 is None and va["acc"] >= 90.0:
            ep90 = ep
    return {"init_val_acc": init_val["acc"], "best_val": best_val, "test_at_best_val": best_test, "final_test": te["acc"], "epochs_to_90": ep90, "hist": hist}


def stage_transplant(args, data, device):
    out_f = OUT / "transplant.json"
    done = json.load(open(out_f)) if out_f.is_file() else {}
    sseeds = [int(s) for s in args.student_seeds.split(",")]
    for seed in [int(s) for s in args.transplant_teacher_seeds.split(",")]:
        d = OUT / f"teacher_s{seed}"
        jobs = [("scratch", None, None)] + [(kind, f, d / (f"ck_{int(f * 100):03d}.pt" if kind == "raw" else f"ref_{int(f * 100):03d}.pt")) for f in FRACTIONS for kind in ("raw", "refined")]
        for kind, f, path in jobs:
            for ss in sseeds:
                key = f"t{seed}_{kind}_{'none' if f is None else int(f * 100):03d}_s{ss}" if f is not None else f"t{seed}_scratch_s{ss}"
                if key in done:
                    continue
                t0 = time.time()
                state = torch.load(path, map_location=device) if path is not None else None
                r = train_student(state, data, device, ss, args.student_epochs)
                done[key] = {"teacher_seed": seed, "kind": kind, "fraction": f, "student_seed": ss, **r, "wall_s": round(time.time() - t0, 1)}
                json.dump(done, open(out_f, "w"), indent=2)
                print(json.dumps({"key": key, "init": round(r["init_val_acc"], 2), "test@bv": round(r["test_at_best_val"], 2), "ep90": r["epochs_to_90"], "s": done[key]["wall_s"]}), flush=True)


def main():
    args = parse()
    global OUT, FRACTIONS, BASIN_FRACTIONS
    OUT = Path(args.out)
    if args.fractions:
        FRACTIONS = [float(f) for f in args.fractions.split(",")]
        BASIN_FRACTIONS = [f for f in BASIN_FRACTIONS if f in FRACTIONS] or FRACTIONS[:2]
    device = get_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data(device, args.probe_size)
    print(json.dumps({"device": str(device), "steps_per_epoch": data.steps_per_epoch, "norm_mean": data.mean, "norm_std": data.std}), flush=True)
    stages = ["checkpoints", "refine", "basins", "transplant"] if args.stage == "all" else args.stage.split(",")
    for st in stages:
        t0 = time.time()
        {"checkpoints": stage_checkpoints, "refine": stage_refine, "basins": stage_basins, "transplant": stage_transplant}[st](args, data, device)
        print(json.dumps({"stage_done": st, "wall_min": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    main()
