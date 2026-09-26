"""Ridge saddle between two teacher minima via the string method + climbing image, then
Newton refinement, Lanczos verification and a transplant test (neurips/08_ridge_saddle_neb.md).

For each teacher pair (A, B), both trained to a minimum with the normalised-input recipe:
  1. string method on the training loss (fixed NEB_IMAGES-image train subset): K interior
     nodes initialised on the straight line A->B, perpendicular-gradient descent with
     equal-arclength redistribution; then climbing-image ascent along the path tangent for
     the highest node;
  2. refine_to_stationary from the climbing image (Newton/MINRES, fixed tolerance) on a
     HESS_IMAGES-image subset; Lanczos extremes (k=5) and SLQ index estimate at the result;
  3. losses of endpoints, path nodes and the refined point on the full train set and the
     validation set; barrier = max path loss minus endpoint mean;
  4. transplant into the student: refined ridge saddle, raw endpoint minimum (A), A's 35%
     checkpoint, scratch; 3 student seeds x 30 epochs; test accuracy at best val and
     epochs to 90% val.

    python neb_ridge_saddle.py --pairs 0:1,2:3
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

from common import get_device, set_seed
from saddle_timing_study import Data, evaluate, flat, grad_norm_on, hessian_stats, load_flat, train_student

HERE = Path(__file__).resolve().parent
ST = HERE / "results" / "saddle_timing"
OUT = HERE / "results" / "neb_ridge"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="0:1,2:3")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--nodes", type=int, default=12, help="interior nodes")
    p.add_argument("--neb-images", type=int, default=4096)
    p.add_argument("--hess-images", type=int, default=1024)
    p.add_argument("--string-iters", type=int, default=300)
    p.add_argument("--climb-iters", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--max-step", type=float, default=0.5, help="clip the per-node update norm (string and climbing phases)")
    p.add_argument("--refine-max-steps", type=int, default=60)
    p.add_argument("--refine-grad-tol", type=float, default=1e-2)
    p.add_argument("--student-epochs", type=int, default=30)
    p.add_argument("--student-seeds", default="0,1,2")
    p.add_argument("--device", default=None)
    return p.parse_args()


# --------------------------------------------------------------------------- teachers

def get_teacher(seed, args, data, device):
    """40-epoch teacher; reuse the saddle-timing checkpoints when present."""
    from sprkd.models import MalariaTeacherCNN
    t = MalariaTeacherCNN().to(device)
    ck = ST / f"teacher_s{seed}" / "ck_100.pt"
    if ck.is_file():
        t.load_state_dict(torch.load(ck, map_location=device)); return t, "saddle_timing ck_100"
    d = OUT / f"teacher_s{seed}"; d.mkdir(parents=True, exist_ok=True)
    if (d / "final.pt").is_file():
        t.load_state_dict(torch.load(d / "final.pt", map_location=device)); return t, "cached"
    set_seed(seed); t = MalariaTeacherCNN().to(device)
    opt = torch.optim.Adam(t.parameters(), 1e-3); batches = data.train_batches(seed)
    for _ in range(args.epochs * data.steps_per_epoch):
        t.train(); x, y = next(batches); opt.zero_grad(); F.cross_entropy(t(x), y).backward(); opt.step()
    torch.save(t.state_dict(), d / "final.pt")
    return t, "trained"


# --------------------------------------------------------------------------- functional loss on flat vectors

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

    def loss(self, v):
        return F.cross_entropy(functional_call(self.model, self.unflat(v), (self.x,)), self.y)

    def loss_and_grad(self, v):
        v = v.detach().requires_grad_(True)
        l = self.loss(v)
        g, = torch.autograd.grad(l, v)
        return float(l), g.detach()

    @torch.no_grad()
    def loss_batched(self, v, x, y, bs=2048):
        tot = 0.0
        for i in range(0, len(y), bs):
            tot += float(F.cross_entropy(functional_call(self.model, self.unflat(v), (x[i:i + bs],)), y[i:i + bs], reduction="sum"))
        return tot / len(y)


def redistribute(nodes):
    """Equal-arclength reparametrisation of the piecewise-linear path (string method)."""
    seg = torch.stack([(nodes[i + 1] - nodes[i]).norm() for i in range(len(nodes) - 1)])
    cum = torch.cat([torch.zeros(1, device=seg.device), torch.cumsum(seg, 0)])
    total = cum[-1]
    targets = torch.linspace(0, float(total), len(nodes), device=seg.device)
    new = [nodes[0]]
    for t in targets[1:-1]:
        j = int(torch.searchsorted(cum, t, right=True).item()) - 1
        j = max(0, min(j, len(nodes) - 2))
        a = (t - cum[j]) / max(float(seg[j]), 1e-12)
        new.append(nodes[j] + a * (nodes[j + 1] - nodes[j]))
    new.append(nodes[-1])
    return new


def upwind_tangent(nodes, losses, i):
    lp, l0, ln = losses[i - 1], losses[i], losses[i + 1]
    tp, tn = nodes[i] - nodes[i - 1], nodes[i + 1] - nodes[i]
    if ln > l0 > lp:
        tau = tn
    elif ln < l0 < lp:
        tau = tp
    else:
        dmax, dmin = max(abs(ln - l0), abs(lp - l0)), min(abs(ln - l0), abs(lp - l0))
        tau = tn * dmax + tp * dmin if ln > lp else tn * dmin + tp * dmax
    return tau / (tau.norm() + 1e-12)


def string_method(fl, a, b, args, log):
    K = args.nodes
    nodes = [a + (b - a) * (i / (K + 1)) for i in range(K + 2)]
    hist = []
    for it in range(args.string_iters):
        losses, grads = [], []
        for v in nodes:
            l, g = fl.loss_and_grad(v); losses.append(l); grads.append(g)
        if any(math.isnan(l) for l in losses):
            raise RuntimeError(f"string method diverged at iteration {it}: nan loss")
        for i in range(1, K + 1):
            tau = upwind_tangent(nodes, losses, i)
            g_perp = grads[i] - torch.dot(grads[i], tau) * tau
            step = args.lr * g_perp
            sn = float(step.norm())
            if sn > args.max_step:
                step = step * (args.max_step / sn)
            nodes[i] = nodes[i] - step
        if it % 10 == 9:
            nodes = redistribute(nodes)
        if it % 25 == 0 or it == args.string_iters - 1:
            hist.append({"phase": "string", "iter": it, "max_loss": max(losses), "argmax": int(max(range(len(losses)), key=lambda j: losses[j])), "losses": [round(l, 4) for l in losses]})
            log(hist[-1])
    # climbing image: highest interior node ascends along the tangent, descends perpendicular
    losses = [fl.loss_and_grad(v)[0] for v in nodes]
    ci = max(range(1, K + 1), key=lambda j: losses[j])
    for it in range(args.climb_iters):
        losses = [fl.loss_and_grad(v)[0] for v in nodes]
        l, g = fl.loss_and_grad(nodes[ci])
        tau = upwind_tangent(nodes, losses, ci)
        force = -(g - 2.0 * torch.dot(g, tau) * tau)          # -g_perp + g_par
        step = args.lr * force
        sn = float(step.norm())
        if sn > args.max_step:
            step = step * (args.max_step / sn)
        nodes[ci] = nodes[ci] + step
        if it % 50 == 0 or it == args.climb_iters - 1:
            gn_perp = float((g - torch.dot(g, tau) * tau).norm())
            hist.append({"phase": "climb", "iter": it, "ci": ci, "loss": l, "grad_norm": float(g.norm()), "grad_perp_norm": gn_perp})
            log(hist[-1])
    return nodes, ci, hist


# --------------------------------------------------------------------------- main

def main():
    args = parse()
    device = get_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    data = Data(device, 256)
    from sprkd.models import MalariaTeacherCNN
    from sprkd.hessian_utils import extreme_eigenpairs
    from sprkd.saddle import refine_to_stationary
    ce = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(777)
    perm = torch.randperm(len(data.y_tr), generator=g)
    neb_idx = perm[:args.neb_images].to(device); hess_idx = perm[:args.hess_images]
    x_neb, y_neb = data.x_tr[neb_idx], data.y_tr[neb_idx]
    hess_batch = (data.x_tr[hess_idx.to(device)].cpu(), data.y_tr[hess_idx.to(device)].cpu())
    log_f = open(OUT / "neb_log.jsonl", "a")

    def log(rec):
        print(json.dumps(rec), flush=True); log_f.write(json.dumps(rec) + "\n"); log_f.flush()

    results = json.load(open(OUT / "results.json")) if (OUT / "results.json").is_file() else {}
    for pair in args.pairs.split(","):
        sa, sb = [int(v) for v in pair.split(":")]
        key = f"pair_{sa}_{sb}"
        if key in results and "transplant" in results[key]:
            print(f"{key} done"); continue
        t0 = time.time()
        tA, srcA = get_teacher(sa, args, data, device); tB, srcB = get_teacher(sb, args, data, device)
        fl = FlatLoss(tA, x_neb, y_neb)
        a, b = flat(tA).to(device), flat(tB).to(device)
        evA, _ = evaluate(tA, data.x_va, data.y_va); evB, _ = evaluate(tB, data.x_va, data.y_va)
        rec = {"pair": [sa, sb], "teacher_sources": [srcA, srcB], "endpoint_val": [evA, evB],
               "endpoint_train_loss_full": [fl.loss_batched(a, data.x_tr, data.y_tr), fl.loss_batched(b, data.x_tr, data.y_tr)],
               "endpoint_dist_l2": float((a - b).norm()), "neb_images": args.neb_images}
        log({"pair": key, "endpoints": rec["endpoint_val"], "train_loss": rec["endpoint_train_loss_full"], "l2": rec["endpoint_dist_l2"]})
        # linear path for reference
        lin = [fl.loss_batched(a + float(t) * (b - a), data.x_tr, data.y_tr) for t in torch.linspace(0, 1, 11)]
        rec["linear_path_train_loss"] = lin; rec["linear_barrier_train"] = max(lin) - 0.5 * (lin[0] + lin[-1])
        # string + climbing image
        nodes, ci, hist = string_method(fl, a, b, args, log)
        rec["neb_history"] = hist
        path_train = [fl.loss_batched(v, data.x_tr, data.y_tr) for v in nodes]
        path_val = [fl.loss_batched(v, data.x_va, data.y_va) for v in nodes]
        rec["path_train_loss_full"] = path_train; rec["path_val_loss"] = path_val; rec["climbing_index"] = ci
        rec["neb_barrier_train"] = max(path_train) - 0.5 * (path_train[0] + path_train[-1])
        rec["neb_barrier_val"] = max(path_val) - 0.5 * (path_val[0] + path_val[-1])
        m_ci = MalariaTeacherCNN().to(device); load_flat(m_ci, nodes[ci].cpu())
        ev_ci, _ = evaluate(m_ci, data.x_va, data.y_va)
        rec["climb_point"] = {"val": ev_ci, "train_loss_full": path_train[ci], "grad_norm_hess_batch": grad_norm_on(copy.deepcopy(m_ci).cpu(), *hess_batch)}
        torch.save(m_ci.state_dict(), OUT / f"{key}_climb.pt")
        log({"pair": key, "neb_barrier_train": rec["neb_barrier_train"], "linear_barrier_train": rec["linear_barrier_train"], "climb": rec["climb_point"]})
        # refine to a stationary point on the Hessian batch
        m_ref = copy.deepcopy(m_ci).cpu()
        t1 = time.time()
        rf = refine_to_stationary(m_ref, ce, hess_batch, max_steps=args.refine_max_steps, grad_tol=args.refine_grad_tol, solver_iters=60)
        rf["wall_s"] = round(time.time() - t1, 1)
        ext = extreme_eigenpairs(m_ref, ce, hess_batch, k=5, tol=1e-2, max_iter=20)
        hs = hessian_stats(m_ref, hess_batch, k=5, do_density=True)
        m_ref_dev = copy.deepcopy(m_ref).to(device); v_ref = flat(m_ref).to(device)
        ev_ref, _ = evaluate(m_ref_dev, data.x_va, data.y_va)
        rec["refined"] = {"refine": rf, "lambda_max": ext["lambda_max"], "lambda_min_k": ext["lambda_min_k"], "grad_norm_hess_batch": ext["grad_norm"],
                          "grad_norm_full_train": grad_norm_on(copy.deepcopy(m_ref).cpu(), data.x_tr.cpu(), data.y_tr.cpu()),
                          "index_estimate": hs.get("index_estimate"), "neg_spectral_mass": hs.get("neg_spectral_mass"), "n_below_tau": hs["n_below_tau"],
                          "trace": hs["trace_hutchinson"], "train_loss_full": fl.loss_batched(v_ref, data.x_tr, data.y_tr), "val": ev_ref,
                          "dist_from_climb_rel": float((v_ref - nodes[ci]).norm() / nodes[ci].norm()),
                          "dist_to_A_rel": float((v_ref - a).norm() / (a - b).norm()), "dist_to_B_rel": float((v_ref - b).norm() / (a - b).norm())}
        rec["refined"]["barrier_train_vs_endpoints"] = rec["refined"]["train_loss_full"] - 0.5 * sum(rec["endpoint_train_loss_full"])
        torch.save(m_ref.state_dict(), OUT / f"{key}_refined.pt")
        log({"pair": key, "refined": {k: v for k, v in rec["refined"].items() if k != "refine"}, "refine": rf})
        results[key] = rec; json.dump(results, open(OUT / "results.json", "w"), indent=2)
        # transplant
        arms = {"ridge_saddle": m_ref.state_dict(), "endpoint_min_A": tA.state_dict(), "ck35_A": None, "scratch": None}
        ck35 = ST / f"teacher_s{sa}" / "ck_035.pt"
        if ck35.is_file():
            arms["ck35_A"] = torch.load(ck35, map_location=device)
        else:
            del arms["ck35_A"]
        tr = {}
        for arm, state in arms.items():
            for ss in [int(v) for v in args.student_seeds.split(",")]:
                r = train_student(None if state is None else {k: v.to(device) for k, v in state.items()}, data, device, ss, args.student_epochs)
                tr[f"{arm}_s{ss}"] = {"arm": arm, "student_seed": ss, **{k: v for k, v in r.items() if k != "hist"}}
                log({"pair": key, "transplant": tr[f"{arm}_s{ss}"]})
        rec["transplant"] = tr; rec["wall_s"] = round(time.time() - t0, 1)
        results[key] = rec; json.dump(results, open(OUT / "results.json", "w"), indent=2)
        log({"pair": key, "done": True, "wall_s": rec["wall_s"]})


if __name__ == "__main__":
    main()
