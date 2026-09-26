"""Plots and tables for neurips/07_saddle_timing_analysis.md from results/saddle_timing/."""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "saddle_timing"
FIGS = HERE.parents[0] / "figs"
FIGS.mkdir(exist_ok=True)
LN2 = 0.6931


def load_teachers():
    out = {}
    for d in sorted(RES.glob("teacher_s*")):
        seed = int(d.name.split("_s")[1])
        rec = {}
        for name in ("checkpoints", "metrics", "refined", "basins"):
            f = d / f"{name}.json"
            if f.is_file():
                rec[name] = json.load(open(f))
        out[seed] = rec
    return out


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return float("nan"), float("nan")
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def reclassify(r):
    """Class of the refined point, honouring convergence: unconverged points are not saddles."""
    rf = r["refine"]; lmin = r["lambda_min_after"]; acc = r["refined_val"]["acc"]; loss = r["refined_loss_probe"]
    if not rf["converged"]:
        return "unconverged_negcurv" if lmin < 0 else "unconverged"
    if lmin >= -0.1:
        return "minimum_or_flat"
    if abs(loss - LN2) <= 0.02 * LN2 or acc < 55.0:
        return "trivial_saddle"
    if acc > 70.0:
        return "informative_saddle"
    return "weak_saddle"


def core_table(T):
    rows = []
    fracs = sorted({m["fraction"] for t in T.values() for m in t.get("metrics", [])})
    for f in fracs:
        ms = [m for t in T.values() for m in t.get("metrics", []) if m["fraction"] == f]
        rs = [r for t in T.values() for r in t.get("refined", []) if r["fraction"] == f]
        classes = [reclassify(r) for r in rs]
        rows.append({
            "fraction": f,
            "val_acc": mean_sd([m["val"]["acc"] for m in ms]), "val_loss": mean_sd([m["val"]["loss"] for m in ms]),
            "grad_norm_probe": mean_sd([m["probe"]["grad_norm_probe"] for m in ms]), "lambda_max": mean_sd([m["probe"]["lambda_max"] for m in ms]),
            "lambda_min": mean_sd([m["probe"]["lambda_min_k"][0] for m in ms]), "n_below_tau": mean_sd([m["probe"]["n_below_tau"] for m in ms]),
            "index_est": mean_sd([m["probe"].get("index_estimate") for m in ms]), "trace": mean_sd([m["probe"]["trace_hutchinson"] for m in ms]),
            "val_lambda_min": mean_sd([m["val"]["lambda_min_k"][0] for m in ms]),
            "ref_loss": mean_sd([r["refined_loss_probe"] for r in rs]), "ref_val_acc": mean_sd([r["refined_val"]["acc"] for r in rs]),
            "ref_lambda_min": mean_sd([r["lambda_min_after"] for r in rs]), "ref_gn": mean_sd([r["refine"]["grad_norm_after"] for r in rs]),
            "dist_rel": mean_sd([r["dist_rel"] for r in rs]), "dist_l2": mean_sd([r["dist_l2"] for r in rs]), "steps": mean_sd([r["refine"]["steps"] for r in rs]),
            "converged": sum(1 for r in rs if r["refine"]["converged"]), "n": len(rs),
            "classes": {c: classes.count(c) for c in sorted(set(classes))},
        })
    return rows


def fmt(ms, nd=3):
    m, s = ms
    return f"{m:.{nd}f} +- {s:.{nd}f}"


def write_core_md(rows, path):
    md = ["| fraction | val acc | grad norm (probe) | lambda_max | lambda_min | # < -0.1 (of 5) | index est. | -> refined loss | refined val acc | refined lambda_min | moved (rel) | steps (conv/n) | class |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['fraction']:.2f} | {fmt(r['val_acc'], 2)} | {fmt(r['grad_norm_probe'])} | {fmt(r['lambda_max'], 1)} | {fmt(r['lambda_min'])} | {fmt(r['n_below_tau'], 1)} | {fmt(r['index_est'], 0)} | "
                  f"{fmt(r['ref_loss'])} | {fmt(r['ref_val_acc'], 2)} | {fmt(r['ref_lambda_min'])} | {fmt(r['dist_rel'])} | {fmt(r['steps'], 1)} ({r['converged']}/{r['n']}) | {r['classes']} |")
    Path(path).write_text("\n".join(md) + "\n")
    return "\n".join(md)


def plot_core(rows):
    x = [r["fraction"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].errorbar(x, [r["ref_loss"][0] for r in rows], yerr=[r["ref_loss"][1] for r in rows], marker="o", label="refined (stationary) loss, probe")
    ax[0].errorbar(x, [r["val_loss"][0] for r in rows], yerr=[r["val_loss"][1] for r in rows], marker="s", label="raw checkpoint val loss")
    ax[0].axhline(LN2, ls="--", c="gray", label="ln 2"); ax[0].set_xscale("log"); ax[0].set_xlabel("fraction of training"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=7)
    ax[1].errorbar(x, [r["lambda_min"][0] for r in rows], yerr=[r["lambda_min"][1] for r in rows], marker="o", label="raw checkpoint")
    ax[1].errorbar(x, [r["ref_lambda_min"][0] for r in rows], yerr=[r["ref_lambda_min"][1] for r in rows], marker="s", label="refined point")
    ax[1].axhline(-0.1, ls="--", c="gray", label="-tau"); ax[1].set_xscale("log"); ax[1].set_xlabel("fraction of training"); ax[1].set_ylabel("lambda_min (probe)"); ax[1].legend(fontsize=7)
    ax[2].errorbar(x, [r["ref_val_acc"][0] for r in rows], yerr=[r["ref_val_acc"][1] for r in rows], marker="s", label="refined point")
    ax[2].errorbar(x, [r["val_acc"][0] for r in rows], yerr=[r["val_acc"][1] for r in rows], marker="o", label="raw checkpoint")
    ax[2].set_xscale("log"); ax[2].set_xlabel("fraction of training"); ax[2].set_ylabel("val accuracy (%)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "saddle_timing_core.png", dpi=140); plt.close(fig)


def basins_table(T):
    agg = defaultdict(lambda: defaultdict(list))
    for t in T.values():
        for key, b in t.get("basins", {}).items():
            agg[(b["fraction"], b["kind"])]["disagreement"].append(b["mean_disagreement"])
            agg[(b["fraction"], b["kind"])]["barrier"].append(b["mean_barrier"])
            agg[(b["fraction"], b["kind"])]["max_barrier"].append(b["max_barrier"])
            agg[(b["fraction"], b["kind"])]["l2"].append(b["mean_l2"])
            agg[(b["fraction"], b["kind"])]["end_acc"].append(st.mean(b["end_val_acc"]))
    rows = []
    for (f, kind), v in sorted(agg.items()):
        rows.append({"fraction": f, "kind": kind, **{k: mean_sd(vals) for k, vals in v.items()}, "n": len(v["barrier"])})
    return rows


def write_basins_md(rows, path):
    md = ["| fraction | start point | end val acc | disagreement rate | mean loss barrier (val) | max barrier | mean L2 between endpoints | n teachers |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['fraction']:.2f} | {r['kind']} | {fmt(r['end_acc'], 2)} | {fmt(r['disagreement'], 4)} | {fmt(r['barrier'], 4)} | {fmt(r['max_barrier'], 4)} | {fmt(r['l2'], 3)} | {r['n']} |")
    Path(path).write_text("\n".join(md) + "\n")
    return "\n".join(md)


def plot_basins(rows):
    fracs = sorted({r["fraction"] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    w = 0.35
    for i, (metric, lab) in enumerate((("barrier", "mean loss barrier (val)"), ("disagreement", "prediction disagreement"))):
        for j, kind in enumerate(("raw", "refined")):
            vals = [next((r[metric] for r in rows if r["fraction"] == f and r["kind"] == kind), (float("nan"), 0)) for f in fracs]
            ax[i].bar([k + (j - 0.5) * w for k in range(len(fracs))], [v[0] for v in vals], w, yerr=[v[1] for v in vals], label=kind, capsize=2)
        ax[i].set_xticks(range(len(fracs))); ax[i].set_xticklabels([f"{f:.2f}" for f in fracs]); ax[i].set_xlabel("fraction of training at start point"); ax[i].set_ylabel(lab); ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "saddle_timing_basins.png", dpi=140); plt.close(fig)


def transplant_table():
    f = RES / "transplant.json"
    if not f.is_file():
        return [], ""
    d = json.load(open(f))
    agg = defaultdict(list)
    for r in d.values():
        key = (r["kind"], r["fraction"] if r["fraction"] is not None else -1)
        agg[key].append(r)
    rows = []
    for (kind, frac), rs in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        rows.append({"kind": kind, "fraction": frac, "n": len(rs), "init_val": mean_sd([r["init_val_acc"] for r in rs]),
                     "test": mean_sd([r["test_at_best_val"] for r in rs]), "final_test": mean_sd([r["final_test"] for r in rs]),
                     "ep90": mean_sd([r["epochs_to_90"] for r in rs if r["epochs_to_90"] is not None]), "n_reached_90": sum(1 for r in rs if r["epochs_to_90"] is not None)})
    md = ["| init | fraction | n runs | init val acc | test acc at best val | final test acc | epochs to 90% val (mean, n reached) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        frac = "-" if r["fraction"] < 0 else f"{r['fraction']:.2f}"
        md.append(f"| {r['kind']} | {frac} | {r['n']} | {fmt(r['init_val'], 1)} | {fmt(r['test'], 2)} | {fmt(r['final_test'], 2)} | {fmt(r['ep90'], 1)} ({r['n_reached_90']}) |")
    (FIGS / "saddle_timing_transplant.md").write_text("\n".join(md) + "\n")
    return rows, "\n".join(md)


def plot_transplant(rows):
    if not rows:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for kind, mk in (("raw", "o"), ("refined", "s")):
        rs = [r for r in rows if r["kind"] == kind]
        ax[0].errorbar([r["fraction"] for r in rs], [r["test"][0] for r in rs], yerr=[r["test"][1] for r in rs], marker=mk, label=kind)
        ax[1].errorbar([r["fraction"] for r in rs], [r["ep90"][0] for r in rs], yerr=[r["ep90"][1] for r in rs], marker=mk, label=kind)
    sc = next((r for r in rows if r["kind"] == "scratch"), None)
    if sc:
        ax[0].axhline(sc["test"][0], ls="--", c="gray", label="scratch"); ax[1].axhline(sc["ep90"][0], ls="--", c="gray", label="scratch")
    for a, lab in zip(ax, ("test acc at best val (%)", "epochs to 90% val")):
        a.set_xscale("log"); a.set_xlabel("fraction of teacher training"); a.set_ylabel(lab); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGS / "saddle_timing_transplant.png", dpi=140); plt.close(fig)


def main():
    T = load_teachers()
    rows = core_table(T)
    if rows:
        print(write_core_md(rows, FIGS / "saddle_timing_core.md")); plot_core(rows)
    br = basins_table(T)
    if br:
        print(); print(write_basins_md(br, FIGS / "saddle_timing_basins.md")); plot_basins(br)
    tr, md = transplant_table()
    if tr:
        print(); print(md); plot_transplant(tr)


if __name__ == "__main__":
    main()
