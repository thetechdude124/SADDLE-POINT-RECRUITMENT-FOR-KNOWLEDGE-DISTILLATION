"""Timing probe for Hessian work on the real target device, before any paid stage.

Loads one real S1 teacher checkpoint and the S1 probe batch, asserts the device, and times:
one Hessian-vector product, one curvature() call (Lanczos extremes), N Adam refinement
steps, and M HiSD steps. Writes probe_timing.json into --out-root/<run-name>/ and prints a
per-stage projection for the S1 refine configuration (9 checkpoints x refine-steps).

Exists because on 2026-09-28 the refine stage ran on the CPU for hours and nobody had timed
one step (neurips/10_s1_incident.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=".")
    p.add_argument("--out-root", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--saddle-dir", required=True, help="dir holding probe.pt and teacher_s<seed>/ck_<frac>.pt")
    p.add_argument("--teacher", default="resnet32x4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", default="050")
    p.add_argument("--probe-size", type=int, default=256)
    p.add_argument("--refine-steps", type=int, default=20)
    p.add_argument("--hisd-steps", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=0)
    a, _ = p.parse_known_args()

    from models import build_model
    from sprkd.hessian_utils import HessianOperator
    from sprkd.saddle import refine_to_stationary
    import cifar_saddle_study as cs

    dev = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.environ.get("SPRKD_REQUIRE_CUDA") == "1" and dev.type != "cuda":
        raise SystemExit(f"SPRKD_REQUIRE_CUDA is set but the device is {dev}")
    cs.DEVICE = dev
    out = Path(a.out_root) / a.run_name; out.mkdir(parents=True, exist_ok=True)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    sd = Path(a.saddle_dir)
    px, py = torch.load(sd / "probe.pt")
    px, py = px[: a.probe_size].to(dev), py[: a.probe_size].to(dev)

    def load():
        m = build_model(a.teacher, 100).to(dev)
        m.load_state_dict(torch.load(sd / f"teacher_s{a.seed}" / f"ck_{a.ckpt}.pt", map_location=dev))
        assert next(m.parameters()).device.type == dev.type
        return m

    res = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "teacher": a.teacher, "ckpt": f"s{a.seed}/ck_{a.ckpt}", "probe_size": int(px.shape[0])}

    m = load(); m.eval()
    op = HessianOperator(m, nn.CrossEntropyLoss(), (px, py))
    v = torch.randn(op.n, device=dev)
    op.hvp(v); sync()                                   # warm-up
    t0 = time.time()
    for _ in range(5):
        op.hvp(v)
    sync(); res["hvp_s"] = round((time.time() - t0) / 5, 4)
    res["n_params"] = op.n
    op.release()
    print(json.dumps({"hvp_s": res["hvp_s"], "device": res["device"]}), flush=True)

    t0 = time.time(); c = cs.curvature(load(), (px, py)); sync()
    res["curvature_s"] = round(time.time() - t0, 2); res["curvature_n_hvp"] = c["n_hvp"]
    res["lambda_max"] = c["lambda_max"]; res["lambda_min"] = c["lambda_min_k"][0]
    print(json.dumps({"curvature_s": res["curvature_s"], "n_hvp": c["n_hvp"]}), flush=True)

    for method, steps, key in (("adam", a.refine_steps, "adam"), ("hisd", a.hisd_steps, "hisd")):
        m = load()
        t0 = time.time()
        rec = refine_to_stationary(m, nn.CrossEntropyLoss(), (px, py), max_steps=steps, grad_tol=0.0, lr=1e-3 if method == "adam" else 1e-2,
                                   method=method, hisd_index=1,
                                   progress_cb=lambda s, g, e, k=key, n=steps: print(json.dumps({"progress": s, "of": n, "elapsed_s": round(e, 1), "stage": k}), flush=True))
        sync()
        wall = time.time() - t0
        res[f"{key}_steps"] = rec["steps"]; res[f"{key}_wall_s"] = round(wall, 2)
        res[f"{key}_s_per_step"] = round(wall / max(rec["steps"], 1), 4)
        res[f"{key}_n_hvp"] = rec["n_hvp"]
        assert next(m.parameters()).device.type == dev.type, "refinement moved the model off the device"

    # projections for the S1 configuration (9 checkpoints x 3000 Adam steps; HiSD 9 x 2000)
    res["proj_refine_adam_h_per_container"] = round(9 * 3000 * res["adam_s_per_step"] / 3600, 2)
    res["proj_hisd_h_per_container"] = round(9 * 2000 * res["hisd_s_per_step"] / 3600, 2)
    res["proj_curvature_h_per_container"] = round(9 * res["curvature_s"] / 3600, 3)
    (out / "probe_timing.json").write_text(json.dumps(res, indent=2))
    (out / "summary.json").write_text(json.dumps({"final_test_top1": None, "probe": res}, indent=2))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
