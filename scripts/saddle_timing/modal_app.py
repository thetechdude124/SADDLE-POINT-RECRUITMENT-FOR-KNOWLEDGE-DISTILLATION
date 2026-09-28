"""Modal entry point for the full experiment matrix (CIFAR-100, TinyImageNet, ImageNet-100).

The same `train.py` / `sprkd_runner.py` scripts that run locally on MPS/CPU run here on a
cloud GPU: the bench directory and the `sprkd` package are mounted into the image and a
persistent Volume (`sprkd-bench`) holds datasets, teacher checkpoints, per-run logs and
results. One container per (config x seed); fan-out via `.map`.

Usage (once `modal token new` points at the workspace with credits):

    # inspect a matrix and its cost estimate; launches nothing
    modal run neurips/bench/modal_app.py::sweep --matrix all --dry-run

    # correctness smoke on real data (5 epochs, 4 arms)
    modal run neurips/bench/modal_app.py::sweep --matrix pilot --seeds 0 --gpu A10G

    # the lean gate on T4 with a cost guard
    modal run --detach neurips/bench/modal_app.py::sweep --matrix gate --seeds 0,1,2 --max-cost-usd 40

    # one-off data preparation for ImageNet-100 (Hugging Face clane9/imagenet-100 -> ImageFolder)
    modal run neurips/bench/modal_app.py::prepare_imagenet100

    # pull results back
    modal volume get sprkd-bench results/runs neurips/bench/results/modal_runs
    python neurips/bench/results/aggregate.py --runs neurips/bench/results/modal_runs

Matrices: pilot, s0/s1/s2/s3 (CIFAR-100 saddle study stages, A100), gate (lean first paid launch without Hessian work: smoke + scratch, last-checkpoint init at 20/50/100% of a 2-epoch weak teacher, KD-weak; 3 seeds), e1 (malaria redo), budget (the $300 de-risking plan), block1 (strong-teacher CRD pairs), block2 (weak-teacher regime + teacher
smaller than student), block3 (component/rule ablations), block4 (TinyImageNet),
block5 (ImageNet-100), all. Blocks with a teacher dependency run in two waves (teachers
first, then everything that needs their checkpoints).

`--gpu` selects the accelerator for the whole sweep. Policy: T4 only (default); A10G is
accepted as the fallback if T4 is not offered; anything else is refused.
`--concurrency` caps simultaneous containers (default 32). `--max-cost-usd N` refuses to
launch when the estimator's cost exceeds N. `--driver` runs the wave loop inside a Modal CPU
function (use with `--detach`), so the launch survives the laptop sleeping; progress is
written to the volume under results/logs/driver_<matrix>_*.log.

This module imports without a Modal login or network access: Modal objects are created
lazily; `build_matrix`, `estimate_cost` and `cfg_to_argv` are pure functions with tests.
"""

from __future__ import annotations

import itertools
import json
import os
import io
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

BENCH = Path(__file__).resolve().parent
_local_pkg = BENCH.parents[1] / "Saddle-point-recruitment-for-knowledge-distillation" / "sprkd" if len(BENCH.parents) > 1 else None
# Inside a Modal container this module is imported from /root/modal_app.py; the package is mounted at /root/sprkd_pkg/sprkd.
SPRKD_PKG = _local_pkg if (_local_pkg is not None and _local_pkg.is_dir()) else Path("/root/sprkd_pkg") / "sprkd"
ALLOWED_GPUS = ("T4", "A10G", "A100", "A100-80GB")   # policy (2026-09-27): A-series allowed for the saddle stages; gate stays on T4 by default
GPU_DEFAULT = os.environ.get("SPRKD_GPU", "T4")
GPU_IMAGENET = os.environ.get("SPRKD_GPU_IMAGENET", "T4")
VOL_NAME = "sprkd-bench"
REMOTE_BENCH = "/root/bench"
REMOTE_PKG_PARENT = "/root/sprkd_pkg"
VOL = "/vol"

# Modal list prices, USD per GPU-hour, verified 2026-09-25 (modal.com/pricing; per-second
# billing; region pinning multiplies by 1.5-1.75x). Re-check before budgeting.
GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "A10G": 1.10, "L4": 0.80, "T4": 0.59}
GPU_SPEED_VS_A100 = {"H100": 1.6, "A100-80GB": 1.05, "A100": 1.0, "A10G": 0.45, "L4": 0.35, "T4": 0.2}
# CPU and memory are billed on top of the GPU: train() asks for 8 cores and 32 GiB, roughly
# 8 x $0.047 + 32 x $0.008 per hour at Modal list prices (approximate).
CONTAINER_OVERHEAD_USD_PER_HOUR = 4 * 0.0000131 * 3600 + 16 * 0.00000222 * 3600  # 4 cores + 16 GiB, ~$0.32/h
# Every GPU run gets a hard Modal timeout of TIMEOUT_FACTOR x its estimated hours (clamped), so
# the worst-case bill of a launch is bounded by TIMEOUT_FACTOR x the estimate, and the launch is
# refused if that worst case exceeds --max-cost-usd.
TIMEOUT_FACTOR = 2.0
MIN_TIMEOUT_S, MAX_TIMEOUT_S = 15 * 60, 24 * 3600

# Rough A100 hours per 240-epoch CIFAR-100 run, scaled from the M2 Pro pilot (resnet8x4 at
# 42 s/epoch locally, ~7x faster on A100 with an 8-core data pipeline). TinyImageNet is ~2x
# CIFAR per epoch (100k images at 64x64), ImageNet-100 (~127k images at 224) ~12x.
A100_HOURS_240EP = {"resnet8x4": 0.45, "wrn16_2": 0.30, "vgg8": 0.25, "resnet20": 0.25,
                    "resnet32x4": 1.6, "wrn40_2": 0.7, "vgg13": 0.5, "resnet110": 0.8,
                    "tv_resnet18": 3.0, "tv_resnet50": 8.0}
DATASET_TIME_MULT = {"cifar100": 1.0, "tinyimagenet": 2.0, "imagenet100": 4.0}

PAIRS = [("resnet32x4", "resnet8x4"), ("wrn40_2", "wrn16_2"), ("vgg13", "vgg8"), ("resnet110", "resnet20")]
SMALL_TEACHER_PAIRS = [("resnet20", "resnet8x4"), ("wrn16_2", "wrn40_2")]
RUNNER_SCRIPTS = {"train": "train.py", "sprkd": "sprkd_runner.py", "malaria": "malaria_e1.py", "cifar_saddle": "cifar_saddle_study.py", "control": "control_run.py"}
GPU_SADDLE = os.environ.get("SPRKD_GPU_SADDLE", "A100")
MALARIA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
CRD_TEACHER_FILES = {"resnet32x4": "resnet32x4_vanilla", "wrn40_2": "wrn_40_2_vanilla",
                     "vgg13": "vgg13_vanilla", "resnet110": "resnet110_vanilla"}
TEACHER_STRENGTHS = [2, 10, 30]


# --------------------------------------------------------------------------- #
# Pure helpers (no Modal needed)
# --------------------------------------------------------------------------- #

def _cfg(runner: str, run_name: str, **args) -> dict:
    return {"runner": runner, "run_name": run_name, "args": args}


def _run_ckpt(run_name: str) -> str:
    """Path (on the Volume) of a finished run's best checkpoint; used for wave-2 KD arms."""
    return f"RUN:{run_name}"


def _run_frac_ckpt(run_name: str, fraction: float) -> str:
    """A fractional checkpoint (ck_fXXX.pt) saved by train.py --ckpt-fractions."""
    return f"RUNCK:{run_name}:{int(round(fraction * 100)):03d}"


try:
    import budget as B
except ImportError:  # running from another cwd
    sys.path.insert(0, str(Path(__file__).resolve().parent)); import budget as B


def build_matrix(name: str, seeds: List[int], epochs: int | None = None) -> List[List[dict]]:
    """Return the run configs for a named matrix as a list of waves (each wave a list of
    configs); wave k+1 may depend on checkpoints written by wave k. Config format:
    {"runner": "train"|"sprkd", "run_name": str, "args": {cli flag without '--': value}}."""
    E = epochs
    if name == "all":
        waves: List[List[dict]] = []
        for sub in ("block1", "block2", "block3", "block4", "block5"):
            for i, w in enumerate(build_matrix(sub, seeds, epochs)):
                while len(waves) <= i:
                    waves.append([])
                waves[i] += w
        return waves

    if name == "pilot":  # correctness smoke on real data
        E = E or 5
        t, s = PAIRS[0]
        w = []
        for seed in seeds:
            w.append(_cfg("train", f"pilot_{s}_scratch_s{seed}", mode="scratch", model=s, epochs=E, milestones=[3, 4], seed=seed))
            w.append(_cfg("train", f"pilot_{s}_kd_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt="CRD", epochs=E, milestones=[3, 4], seed=seed))
            w.append(_cfg("sprkd", f"pilot_{s}_sprkd_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=1, saddle_steps=100, epochs=E, milestones=[3, 4], seed=seed))
            w.append(_cfg("sprkd", f"pilot_{s}_asrinit_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=1, saddle_steps=100, epochs=E, milestones=[3, 4], seed=seed, init_only=True))
        return [w]

    if name == "control":  # verifies spawn/record/poll/abort/budget-stop/halt for cents; no GPU work
        return [[_cfg("control", "c_fast", total_steps=60, step_s=1.0),
                 _cfg("control", "c_slow", total_steps=1200, step_s=1.0, report_of=36000),
                 _cfg("control", "c_sleep", total_steps=1200, step_s=1.0)]]
    E = E or 240
    if name == "s0":  # smoke of every cifar_saddle stage on a subset, one container each, sequential waves
        c = dict(subset=2048, teacher="resnet32x4", student="resnet8x4", probe_size=64, out_dir="cifar_saddle_smoke")
        return [[_cfg("cifar_saddle", "s0_teachers", stage="teachers", seeds="0,1", teacher_epochs=1, fractions="0.5,1.0", **c)],
                [_cfg("cifar_saddle", "s0_refine", stage="refine", seeds="0", fractions="0.5,1.0", refine_steps=20, **c),
                 _cfg("cifar_saddle", "s0_hisd", stage="refine", seeds="0", fractions="1.0", refine_method="hisd", hisd_index=1, refine_steps=20, refine_lr=0.01, **c),
                 _cfg("cifar_saddle", "s0_ridge", stage="ridge", pairs="0:1", nodes=3, neb_images=256, string_iters=5, climb_iters=5, refine_steps=20, **c)],
                [_cfg("cifar_saddle", "s0_transplant", stage="transplant", point="ridge:0:1", student_seeds="0", student_epochs=2, milestones=[1], **c)]]
    if name == "s1":  # do informative saddles exist on CIFAR-100? weak teachers + checkpoint curvature + refinement
        sd = ",".join(str(x) for x in seeds)
        w1 = [_cfg("cifar_saddle", f"s1_teachers_s{seed}", stage="teachers", seeds=str(seed), teacher_epochs=10,
                   fractions="0.02,0.05,0.1,0.2,0.35,0.5,0.65,0.8,1.0", probe_size=256) for seed in seeds]
        w2 = [_cfg("cifar_saddle", f"s1_refine_s{seed}", stage="refine", seeds=str(seed), fractions="0.02,0.05,0.1,0.2,0.35,0.5,0.65,0.8,1.0",
                   refine_method="adam", refine_steps=3000, refine_lr=1e-3, probe_size=256) for seed in seeds]
        # HiSD: climb from the final weights (a minimum) to the nearest index-1 saddle
        w2 += [_cfg("cifar_saddle", f"s1_hisd_s{seed}", stage="refine", seeds=str(seed), fractions="1.0", refine_method="hisd", hisd_index=1,
                    refine_steps=3000, refine_lr=0.01, probe_size=256) for seed in seeds]
        return [w1, w2]
    if name == "s2":  # ridge search between teacher minima (needs s1 teachers)
        prs = [(seeds[i], seeds[(i + 1) % len(seeds)]) for i in range(len(seeds))] if len(seeds) > 1 else []
        return [[_cfg("cifar_saddle", f"s2_ridge_{a}_{b}", stage="ridge", pairs=f"{a}:{b}", nodes=10, neb_images=2048, string_iters=300, climb_iters=300,
                      refine_method="adam", refine_steps=3000, refine_lr=1e-3) for a, b in prs]]
    if name == "s3":  # transplant, full 240-epoch CRD schedule (needs s1 and s2); scratch / KD-weak come from the gate
        pts = [f"ridge:{seeds[i]}:{seeds[(i + 1) % len(seeds)]}" for i in range(len(seeds))] if len(seeds) > 1 else []
        pts += [f"final:{seeds[0]}", f"hisd:{seeds[0]}:1.0", f"ck:{seeds[0]}:0.5", f"ref:{seeds[0]}:0.5", f"ck:{seeds[0]}:0.2", f"ref:{seeds[0]}:0.2"]
        return [[_cfg("cifar_saddle", "s3_" + pt.replace(":", "_"), stage="transplant", point=pt, student_seeds="0,1,2", student_epochs=E, milestones=[150, 180, 210]) for pt in pts]]

    if name == "gate":  # lean first paid launch, no Hessian work (07_saddle_timing_analysis.md):
        # E0 smoke (2 arms, 3 epochs, 1 seed) + per seed: 2-epoch weak teacher with checkpoints at 20/50/100%,
        # scratch, last-checkpoint init at each fraction, KD-weak with the fixed loss. 240-epoch CRD schedule.
        t, s = PAIRS[0]
        w0 = [_cfg("train", f"pilot_{s}_scratch_s{seeds[0]}", mode="scratch", model=s, epochs=3, milestones=[2], seed=seeds[0]),
              _cfg("sprkd", f"pilot_{s}_lastinit_from_{t}_s{seeds[0]}", model=s, teacher=t, teachers=1, teacher_epochs=1, saddle_steps=100000, asr_mode="last", init_only=True, epochs=3, milestones=[2], seed=seeds[0])]
        w1, w2 = [], []
        for seed in seeds:
            tname = f"gate_teacher_{t}_2ep_s{seed}"
            w1.append(_cfg("train", tname, mode="scratch", model=t, epochs=2, milestones=[1], seed=seed, save_ckpt=True, ckpt_fractions=[0.2, 0.5, 1.0]))
            w1.append(_cfg("train", f"gate_{s}_scratch_s{seed}", mode="scratch", model=s, epochs=E, seed=seed))
            for fr in (0.2, 0.5, 1.0):
                w2.append(_cfg("sprkd", f"gate_{s}_lastinit_f{int(fr * 100):03d}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=0,
                               teacher_ckpt=_run_frac_ckpt(tname, fr), asr_mode="last", init_only=True, epochs=E, seed=seed))
            w2.append(_cfg("train", f"gate_{s}_kdweak_T2_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt=_run_ckpt(tname), epochs=E, seed=seed))
        return [w0 + w1, w2]

    if name == "e1":  # malaria redo (one container per seed runs every arm; tiny models)
        return [[_cfg("malaria", f"malaria_e1_s{seed}", seed=seed, arms="all", epochs=epochs or 100) for seed in seeds]]

    if name == "budget":  # the $300-budget de-risking plan (02_evaluation.md, section e), 3 seeds by default
        w1, w2 = [], []
        t, s = PAIRS[0]                       # resnet32x4 -> resnet8x4
        t2, s2 = PAIRS[1]                     # wrn40_2 -> wrn16_2
        for seed in seeds:
            for (tt, ss) in ((t, s), (t2, s2)):   # E2 gate + E3 generality: 7 arms x 2 pairs
                base = dict(model=ss, teacher=tt, teachers=1, teacher_epochs=2, saddle_steps=50, epochs=E, seed=seed)
                w1.append(_cfg("train", f"bud_{ss}_scratch_s{seed}", mode="scratch", model=ss, epochs=E, seed=seed))
                w1.append(_cfg("train", f"bud_{ss}_kd_from_{tt}_s{seed}", mode="kd", model=ss, teacher=tt, teacher_ckpt="CRD", epochs=E, seed=seed))
                w1.append(_cfg("sprkd", f"bud_{ss}_asrinit_T1x2_from_{tt}_s{seed}", **base, init_only=True))
                w1.append(_cfg("sprkd", f"bud_{ss}_sprkd_T1x2_from_{tt}_s{seed}", **base))
                w1.append(_cfg("sprkd", f"bud_{ss}_randsnap_T1x2_from_{tt}_s{seed}", **base, init_only=True, asr_mode="random_snapshot"))
                w1.append(_cfg("sprkd", f"bud_{ss}_lastinit_T1x2_from_{tt}_s{seed}", **base, init_only=True, asr_mode="last"))
                w1.append(_cfg("sprkd", f"bud_{ss}_swainit_T1x2_from_{tt}_s{seed}", **base, init_only=True, asr_mode="swa"))
            # E4 teacher-strength sweep on the resnet pair (2 epochs covered above, 240 = CRD checkpoint)
            for te in (10, 30):
                w1.append(_cfg("train", f"bud_teacher_{t}_{te}ep_s{seed}", mode="scratch", model=t, epochs=te, milestones=[max(1, te - 1)], seed=seed, save_ckpt=True))
                w2.append(_cfg("train", f"bud_{s}_kdweak_T{te}_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt=_run_ckpt(f"bud_teacher_{t}_{te}ep_s{seed}"), epochs=E, seed=seed))
                w2.append(_cfg("sprkd", f"bud_{s}_lastinit_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=0, teacher_ckpt=_run_ckpt(f"bud_teacher_{t}_{te}ep_s{seed}"), asr_mode="last", epochs=E, seed=seed, init_only=True))
                w1.append(_cfg("sprkd", f"bud_{s}_asrinit_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=50, epochs=E, seed=seed, init_only=True))
            w1.append(_cfg("sprkd", f"bud_{s}_wsfinal_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=0, teacher_ckpt="CRD", asr_mode="last", epochs=E, seed=seed, init_only=True))
            # E5 component ablation on the resnet pair
            base = dict(model=s, teacher=t, teachers=1, teacher_epochs=2, saddle_steps=50, epochs=E, seed=seed)
            for v, a in {"noTM": dict(no_tm=True), "noNHE": dict(no_nhe=True), "noPGD": dict(no_pgd=True), "noNHEnoPGD": dict(no_nhe=True, no_pgd=True)}.items():
                w1.append(_cfg("sprkd", f"bud_{s}_sprkd_{v}_from_{t}_s{seed}", **{**base, **a}))
            # E6 teacher smaller than student
            for (tt, ss) in SMALL_TEACHER_PAIRS:
                if ss != s:
                    w1.append(_cfg("train", f"bud_{ss}_scratch_s{seed}", mode="scratch", model=ss, epochs=E, seed=seed))
                w1.append(_cfg("sprkd", f"bud_{ss}_asrinit_T1x10_from_{tt}_s{seed}", model=ss, teacher=tt, teachers=1, teacher_epochs=10, saddle_steps=50, epochs=E, seed=seed, init_only=True))
                w1.append(_cfg("sprkd", f"bud_{ss}_sprkd_T1x10_from_{tt}_s{seed}", model=ss, teacher=tt, teachers=1, teacher_epochs=10, saddle_steps=50, epochs=E, seed=seed))
        return [w1, w2]

    if name == "block1":  # strong CRD teacher available
        w = []
        for (t, s), seed in itertools.product(PAIRS, seeds):
            w.append(_cfg("train", f"b1_{s}_scratch_s{seed}", mode="scratch", model=s, epochs=E, seed=seed))
            w.append(_cfg("train", f"b1_{s}_kd_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt="CRD", epochs=E, seed=seed))
            # weight selection from the strong teacher's final weights (Xu et al. 2024 baseline): ASR-mode "last" on the CRD checkpoint
            w.append(_cfg("sprkd", f"b1_{s}_wsfinal_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=0, teacher_ckpt="CRD", asr_mode="last", epochs=E, seed=seed, init_only=True))
            w.append(_cfg("sprkd", f"b1_{s}_asrinit_T1x2_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=2, saddle_steps=50, epochs=E, seed=seed, init_only=True))
            w.append(_cfg("sprkd", f"b1_{s}_sprkd_T1x2_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=2, saddle_steps=50, epochs=E, seed=seed))
        return [w]

    if name == "block2":  # weak-teacher regime: teacher strength sweep, init controls, KD-weak, small teacher
        w1, w2 = [], []
        for (t, s), te, seed in itertools.product(PAIRS, TEACHER_STRENGTHS, seeds):
            base = dict(model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=50, epochs=E, seed=seed)
            # weak teacher trained once as a plain scratch run (checkpoint reused by KD-weak and WS-final-weak)
            w1.append(_cfg("train", f"b2_teacher_{t}_{te}ep_s{seed}", mode="scratch", model=t, epochs=te, milestones=[max(1, te - 1)], seed=seed, save_ckpt=True))
            w2.append(_cfg("train", f"b2_{s}_kdweak_T{te}_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt=_run_ckpt(f"b2_teacher_{t}_{te}ep_s{seed}"), epochs=E, seed=seed))
            w2.append(_cfg("sprkd", f"b2_{s}_wsfinalweak_T{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=0, teacher_ckpt=_run_ckpt(f"b2_teacher_{t}_{te}ep_s{seed}"), asr_mode="last", epochs=E, seed=seed, init_only=True))
            w2.append(_cfg("sprkd", f"b2_{s}_asrinit_T1x{te}_from_{t}_s{seed}", **base, init_only=True))
            w2.append(_cfg("sprkd", f"b2_{s}_sprkd_T1x{te}_from_{t}_s{seed}", **base))
            w2.append(_cfg("sprkd", f"b2_{s}_randsnap_T1x{te}_from_{t}_s{seed}", **base, init_only=True, asr_mode="random_snapshot"))
            w2.append(_cfg("sprkd", f"b2_{s}_swainit_T1x{te}_from_{t}_s{seed}", **base, init_only=True, asr_mode="swa"))
            for K in (3, 5):
                w2.append(_cfg("sprkd", f"b2_{s}_asrinit_T{K}x{te}_from_{t}_s{seed}", **{**base, "teachers": K}, init_only=True))
        for (t, s), seed in itertools.product(SMALL_TEACHER_PAIRS, seeds):
            w2.append(_cfg("train", f"b2_{s}_scratch_s{seed}", mode="scratch", model=s, epochs=E, seed=seed))
            for te in (10, 30):
                w2.append(_cfg("sprkd", f"b2_{s}_asrinit_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=50, epochs=E, seed=seed, init_only=True))
                w2.append(_cfg("sprkd", f"b2_{s}_sprkd_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=50, epochs=E, seed=seed))
        return [w1, w2]

    if name == "block3":  # component and rule ablations on resnet32x4(2ep) -> resnet8x4
        t, s = PAIRS[0]
        base = dict(model=s, teacher=t, teachers=1, teacher_epochs=2, saddle_steps=50, epochs=E)
        variants = {
            "noTM": dict(no_tm=True), "noNHE": dict(no_nhe=True), "noPGD": dict(no_pgd=True),
            "noNHEnoPGD": dict(no_nhe=True, no_pgd=True),
            "rule-ratio": dict(saddle_rule="ratio"), "rule-both": dict(saddle_rule="both"),
            "eigs2": dict(n_top_eigs=2), "eigs10": dict(n_top_eigs=10),
            "ss10": dict(saddle_steps=10), "ss200": dict(saddle_steps=200),
            "eps1e-1": dict(epsilon=0.1), "eps1e-2": dict(epsilon=0.01),
            "hb256": dict(hessian_batch=256),
        }
        w = [_cfg("sprkd", f"b3_{s}_sprkd_{v}_from_{t}_s{seed}", **{**base, **a}, seed=seed)
             for (v, a), seed in itertools.product(variants.items(), seeds)]
        return [w]

    if name == "block4":  # TinyImageNet: teachers first, then scratch / KD / ASR-init / SPRKD
        E4 = epochs or 100
        pairs = [("resnet32x4", "resnet8x4"), ("wrn40_2", "wrn16_2")]
        w1, w2 = [], []
        for (t, s), seed in itertools.product(pairs, seeds):
            w1.append(_cfg("train", f"b4_teacher_{t}_s{seed}", dataset="tinyimagenet", mode="scratch", model=t, epochs=E4, milestones=[60, 80, 90], seed=seed, save_ckpt=True))
            w2.append(_cfg("train", f"b4_{s}_scratch_s{seed}", dataset="tinyimagenet", mode="scratch", model=s, epochs=E4, milestones=[60, 80, 90], seed=seed))
            w2.append(_cfg("train", f"b4_{s}_kd_from_{t}_s{seed}", dataset="tinyimagenet", mode="kd", model=s, teacher=t, teacher_ckpt=_run_ckpt(f"b4_teacher_{t}_s{seed}"), epochs=E4, milestones=[60, 80, 90], seed=seed))
            for te in (2, 10):
                w2.append(_cfg("sprkd", f"b4_{s}_asrinit_T1x{te}_from_{t}_s{seed}", dataset="tinyimagenet", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=100, epochs=E4, milestones=[60, 80, 90], seed=seed, init_only=True))
                w2.append(_cfg("sprkd", f"b4_{s}_sprkd_T1x{te}_from_{t}_s{seed}", dataset="tinyimagenet", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=100, epochs=E4, milestones=[60, 80, 90], seed=seed))
        return [w1, w2]

    if name == "block5":  # ImageNet-100, torchvision ResNet-50 -> ResNet-18, 100 epochs
        E5 = epochs or 100
        t, s = "tv_resnet50", "tv_resnet18"
        common = dict(dataset="imagenet100", batch_size=128, lr=0.1, wd=1e-4, epochs=E5, milestones=[30, 60, 90], num_workers=12)
        w1, w2 = [], []
        for seed in seeds:
            w1.append(_cfg("train", f"b5_teacher_{t}_s{seed}", mode="scratch", model=t, seed=seed, save_ckpt=True, **common))
            w2.append(_cfg("train", f"b5_{s}_scratch_s{seed}", mode="scratch", model=s, seed=seed, **common))
            w2.append(_cfg("train", f"b5_{s}_kd_from_{t}_s{seed}", mode="kd", model=s, teacher=t, teacher_ckpt=_run_ckpt(f"b5_teacher_{t}_s{seed}"), seed=seed, **common))
            for te in (2, 10):
                w2.append(_cfg("sprkd", f"b5_{s}_asrinit_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=200, hessian_batch=32, seed=seed, init_only=True, **common))
                w2.append(_cfg("sprkd", f"b5_{s}_sprkd_T1x{te}_from_{t}_s{seed}", model=s, teacher=t, teachers=1, teacher_epochs=te, saddle_steps=200, hessian_batch=32, seed=seed, **common))
        return [w1, w2]

    raise KeyError(f"unknown matrix {name!r}; choose pilot, gate, e1, s0, s1, s2, s3, budget, block1, block2, block3, block4, block5, all")


def package_version_info() -> str:
    """Version and git commit of the sprkd package that will be mounted into the image."""
    repo = SPRKD_PKG.parent
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        head, branch = "?", "?"
    ver = "?"
    try:
        for line in open(repo / "pyproject.toml"):
            if line.startswith("version"):
                ver = line.split("=")[1].strip().strip('"')
    except OSError:
        pass
    return f"version {ver}, git {branch}@{head}"


def flatten(waves: List[List[dict]]) -> List[dict]:
    return [c for w in waves for c in w]


GPU_SMOKE = os.environ.get("SPRKD_GPU_SMOKE", "T4")


def gpu_for(cfg: dict, gpu_override: str | None) -> str:
    g = gpu_override or ("T4" if cfg["runner"] == "control" else GPU_SADDLE if cfg["runner"] == "cifar_saddle" else
                         GPU_SMOKE if cfg["run_name"].startswith("pilot_") else
                         GPU_IMAGENET if cfg["args"].get("dataset") == "imagenet100" else GPU_DEFAULT)
    if g not in ALLOWED_GPUS:
        raise ValueError(f"GPU {g!r} is not allowed by the compute policy; use one of {ALLOWED_GPUS}")
    return g


def _gpu_for_unchecked(cfg: dict, gpu_override: str | None) -> str:
    if gpu_override:
        return gpu_override
    if cfg["runner"] == "control":
        return "T4"
    if cfg["runner"] == "cifar_saddle":
        return GPU_SADDLE
    if cfg["run_name"].startswith("pilot_"):
        return GPU_SMOKE
    return GPU_IMAGENET if cfg["args"].get("dataset") == "imagenet100" else GPU_DEFAULT


def run_hours(c: dict, gpu: str | None = None) -> tuple:
    """(gpu, estimated hours) for one config."""
    a = c["args"]; g = _gpu_for_unchecked(c, gpu)
    speed = GPU_SPEED_VS_A100.get(g, 1.0)
    if c["runner"] == "cifar_saddle":
        st = a["stage"]; ep = a.get("student_epochs", 240)
        # NOTE: the refine/ridge figures are guesses, not measurements; the S1 launch of 2026-09-28
        # overran them by more than 10x. Measure one step on the target GPU before trusting them.
        h = {"teachers": 0.8, "refine": 0.9, "ridge": 1.0, "transplant": 3 * 0.45 * ep / 240}[st] * (0.1 if a.get("subset") else 1.0)
        return g, h / speed
    if c["runner"] == "malaria":  # ~8 runs x ~2 min each on A100 for the 6k/25k-param CNNs at 100 epochs
        return g, 0.35 * a.get("epochs", 100) / 100 / speed
    if c["runner"] == "control":
        return g, 0.1
    ds_mult = DATASET_TIME_MULT.get(a.get("dataset", "cifar100"), 1.0)
    h = A100_HOURS_240EP.get(a["model"], 0.5) * a.get("epochs", 240) / 240 * ds_mult
    if c["runner"] == "train" and a.get("mode") == "kd":
        h *= 1.4
    if c["runner"] == "sprkd":
        th = A100_HOURS_240EP.get(a["teacher"], 0.5) * a.get("teacher_epochs", 2) * a.get("teachers", 1) / 240 * ds_mult
        h += th + 0.05 * a.get("teachers", 1)
    return g, h / speed


def run_timeout_s(c: dict, gpu: str | None = None) -> int:
    _, h = run_hours(c, gpu)
    return int(min(MAX_TIMEOUT_S, max(MIN_TIMEOUT_S, TIMEOUT_FACTOR * h * 3600)))


def estimate_cost(configs: List[dict], gpu: str | None = None) -> Dict[str, float]:
    hours_by_gpu: Dict[str, float] = {}
    worst = 0.0
    for c in configs:
        g, h = run_hours(c, gpu)
        rate = GPU_USD_PER_HOUR.get(g, 2.1) + CONTAINER_OVERHEAD_USD_PER_HOUR
        hours_by_gpu[g] = hours_by_gpu.get(g, 0.0) + h
        worst += run_timeout_s(c, gpu) / 3600 * rate
    usd = sum(h * (GPU_USD_PER_HOUR.get(g, 2.1) + CONTAINER_OVERHEAD_USD_PER_HOUR) for g, h in hours_by_gpu.items())
    return {"runs": len(configs), "gpu_hours": {g: round(h, 1) for g, h in hours_by_gpu.items()},
            "usd": round(usd, 0), "worst_case_usd": round(worst, 0)}


def cfg_to_argv(cfg: dict, data_root: str, out_root: str, ckpt_dir: str) -> List[str]:
    script = RUNNER_SCRIPTS[cfg["runner"]]
    if cfg["runner"] == "cifar_saddle":
        out_dir = cfg["args"].get("out_dir", "cifar_saddle")
        argv = [script, "--data-root", data_root, "--out", f"{out_root}/../{out_dir}", "--num-workers", "8"]
    elif cfg["runner"] == "malaria":
        argv = [script, "--data-root", f"{data_root}/cell_images", "--cache", f"{data_root}/malaria_32.pt",
                "--split-file", f"{out_root}/../malaria_split.json", "--out-root", out_root]
    else:
        argv = [script, "--data-root", data_root, "--out-root", out_root, "--run-name", cfg["run_name"]]
        if "num_workers" not in cfg["args"]:
            argv += ["--num-workers", "8"]
    for k, v in cfg["args"].items():
        flag = "--" + k.replace("_", "-")
        if k == "teacher_ckpt":
            if v == "CRD":
                v = str(Path(ckpt_dir) / "crd_teachers" / f"{CRD_TEACHER_FILES[cfg['args']['teacher']]}.pth")
            elif isinstance(v, str) and v.startswith("RUNCK:"):
                _, run, frac = v.split(":")
                v = str(Path(out_root) / run / f"ck_f{frac}.pt")
            elif isinstance(v, str) and v.startswith("RUN:"):
                v = str(Path(out_root) / v[4:] / "best.pt")
        if k == "out_dir":
            continue
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            argv += [flag] + [str(x) for x in v]
        else:
            argv += [flag, str(v)]
    return argv


# --------------------------------------------------------------------------- #
# Modal objects (lazy; importing this module never contacts Modal)
# --------------------------------------------------------------------------- #

try:
    import modal  # type: ignore
except ImportError:  # pragma: no cover
    modal = None

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("curl", "unzip")
        .pip_install("torch>=2.4", "torchvision", "numpy<3", "scipy", "tqdm", "pyhessian", "pytest",
                     "datasets", "pillow", "huggingface_hub")
        .add_local_dir(str(BENCH), remote_path=REMOTE_BENCH,
                       ignore=["data/**", "checkpoints/**", "results/**", "logs/**", "**/__pycache__/**", "**/.pytest_cache/**"])
        .add_local_dir(str(SPRKD_PKG), remote_path=f"{REMOTE_PKG_PARENT}/sprkd", ignore=["**/__pycache__/**"])
    )
    vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
    app = modal.App("sprkd-bench")

    def _ensure_cifar():
        from torchvision import datasets
        root = Path(VOL) / "data"; root.mkdir(parents=True, exist_ok=True)
        if not (root / "cifar-100-python").is_dir():
            datasets.CIFAR100(str(root), train=True, download=True)
            datasets.CIFAR100(str(root), train=False, download=True)
            vol.commit()

    def _ensure_tiny():
        sys.path.insert(0, REMOTE_BENCH)
        from data import ensure_tinyimagenet
        if not (Path(VOL) / "data" / "tiny-imagenet-200" / "train").is_dir():
            ensure_tinyimagenet(Path(VOL) / "data"); vol.commit()

    def _ensure_malaria():
        d = Path(VOL) / "data" / "cell_images"
        if not d.is_dir():
            z = Path(VOL) / "data" / "cell_images.zip"
            subprocess.run(["curl", "-sS", "-L", "-o", str(z), MALARIA_URL], check=True)
            subprocess.run(["unzip", "-q", str(z), "-d", str(Path(VOL) / "data")], check=True)
            vol.commit()

    def _ensure_teachers():
        d = Path(VOL) / "checkpoints" / "crd_teachers"; d.mkdir(parents=True, exist_ok=True)
        changed = False
        for m in CRD_TEACHER_FILES.values():
            f = d / f"{m}.pth"
            if not f.is_file() or f.stat().st_size < 1_000_000:
                subprocess.run(["curl", "-sS", "-L", "-o", str(f), f"http://shape2prog.csail.mit.edu/repo/{m}/ckpt_epoch_240.pth"], check=True)
                changed = True
        if changed:
            vol.commit()

    TRAIN_CPU, TRAIN_MEM_GIB = 4, 16

    @app.function(image=image, gpu=GPU_DEFAULT, volumes={VOL: vol}, timeout=24 * 3600, cpu=TRAIN_CPU, memory=TRAIN_MEM_GIB * 1024)
    def train(cfg: dict) -> dict:
        """Run one config on a GPU. Returns the run's summary.json (plus a log tail on failure)."""
        import traceback
        try:
            return _train_impl(cfg)
        except Exception:
            return {"run_name": cfg.get("run_name"), "returncode": -1, "error": traceback.format_exc()[-3000:]}

    def _train_impl(cfg: dict) -> dict:
        sys.path.insert(0, REMOTE_PKG_PARENT)
        os.chdir(REMOTE_BENCH)
        vol.reload()
        ds = cfg["args"].get("dataset", "cifar100")
        if cfg["runner"] == "malaria":
            _ensure_malaria()
        elif ds == "cifar100":
            _ensure_cifar(); _ensure_teachers()
        elif ds == "tinyimagenet":
            _ensure_tiny()
        out_root = f"{VOL}/results/runs"
        log_dir = Path(VOL) / "results" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
        argv = cfg_to_argv(cfg, f"{VOL}/data", out_root, f"{VOL}/checkpoints")
        env = {**os.environ, "PYTHONPATH": f"{REMOTE_PKG_PARENT}:{REMOTE_BENCH}", "SPRKD_REQUIRE_CUDA": "1"}
        t0 = time.time()
        with open(log_dir / f"{cfg['run_name']}.log", "w") as log:
            proc = subprocess.run([sys.executable] + argv, cwd=REMOTE_BENCH, env=env, stdout=log, stderr=subprocess.STDOUT)
        vol.commit()
        summary_path = Path(out_root) / cfg["run_name"] / "summary.json"
        result = {"run_name": cfg["run_name"], "returncode": proc.returncode, "wall_clock_s": round(time.time() - t0, 1)}
        if summary_path.is_file():
            result["summary"] = json.load(open(summary_path))
        else:
            result["log_tail"] = open(log_dir / f"{cfg['run_name']}.log").read()[-3000:]
            if proc.returncode == 0 and cfg["runner"] == "cifar_saddle":
                result["summary"] = {"stage": cfg["args"]["stage"], "ok": True}
        return result

    @app.function(image=image, volumes={VOL: vol}, timeout=6 * 3600, cpu=8, memory=32768)
    def prepare_imagenet100():
        """Materialise ImageNet-100 (HF `clane9/imagenet-100`) as ImageFolder train/ and val/ on the Volume."""
        from datasets import load_dataset
        root = Path(VOL) / "data" / "imagenet100"
        if (root / "train").is_dir() and (root / "val").is_dir():
            print("imagenet100 already prepared"); return
        ds = load_dataset("clane9/imagenet-100")
        names = ds["train"].features["label"].names
        for split, out in (("train", "train"), ("validation", "val")):
            for i, ex in enumerate(ds[split]):
                d = root / out / names[ex["label"]]; d.mkdir(parents=True, exist_ok=True)
                ex["image"].convert("RGB").save(d / f"{i}.JPEG", quality=95)
                if i % 5000 == 0:
                    print(split, i, flush=True)
        vol.commit()
        print("done:", root)

    LAUNCH_DIR = Path(VOL) / "results" / "launches"
    POLL_S = 30

    def _launch_paths():
        LAUNCH_DIR.mkdir(parents=True, exist_ok=True)
        return sorted(LAUNCH_DIR.glob("*.json"))

    def _save_record(rec: dict, path: Path):
        path.write_text(B.dumps(rec)); vol.commit()

    def _live_record(matrix: str):
        """(record, path) of the newest live launch record for this matrix, or (None, None)."""
        vol.reload()
        for path in reversed(_launch_paths()):
            rec = B.loads(path.read_text())
            if rec["matrix"] == matrix and rec["status"] == B.LIVE:
                return rec, path
        return None, None

    def _cancel_all(rec: dict, reason: str, log):
        """Cancel every live run (terminating its container) and mark the record."""
        for name in B.live_runs(rec):
            r = rec["runs"][name]
            try:
                modal.FunctionCall.from_id(r["call_id"]).cancel(terminate_containers=True)
                B.finish_run(rec, name, "cancelled")
                log(f"cancelled {name} ({r['call_id']}): {reason}")
            except Exception as e:  # keep going; the watchdog retries
                log(f"cancel FAILED for {name} ({r['call_id']}): {e!r}")
        rec["events"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}")

    def _read_run_log(run_name: str) -> str:
        p = Path(VOL) / "results" / "logs" / f"{run_name}.log"
        try:
            return p.read_text()[-20000:]
        except OSError:
            return ""

    def _poll(call_id: str):
        """Result dict, B.PENDING while running (the client raises the built-in TimeoutError for a
        pending call with timeout=0; verified in modal 1.5.5 _functions.poll_function), or raises."""
        try:
            return modal.FunctionCall.from_id(call_id).get(timeout=0)
        except TimeoutError:
            return B.PENDING

    def _cancel(call_id: str):
        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def _run_waves_budgeted(waves, configs, gpu, concurrency, log, rec: dict, rec_path: Path):
        def spawn(c):
            g = gpu_for(c, gpu or None); tmo = run_timeout_s(c, gpu or None)
            call = train.with_options(gpu=g, max_containers=concurrency, timeout=tmo).spawn(c)
            return call.object_id, g, tmo

        def save(r):
            _save_record(r, rec_path)

        def sleep(sec):
            time.sleep(sec); vol.reload()

        return B.run_budgeted(waves, rec, spawn=spawn, poll=_poll, cancel=_cancel, read_log=_read_run_log,
                              save=save, log=log, now=time.time, sleep=sleep, poll_s=POLL_S)

    @app.function(image=image, volumes={VOL: vol}, timeout=24 * 3600, cpu=1, memory=2048)
    def sweep_driver(matrix: str, seeds: List[int], gpu: str = "", epochs: int = 0, concurrency: int = 32, budget_usd: float = 0.0) -> str:
        """Run a whole matrix from inside Modal with a runtime budget. Survives the laptop sleeping;
        after a Modal preemption it re-attaches to the runs in its launch record instead of respawning.
        Progress: /vol/results/logs/driver_<matrix>_<ts>.log; record: /vol/results/launches/<matrix>_<ts>.json."""
        if not budget_usd:
            raise ValueError("sweep_driver requires budget_usd > 0")
        waves = build_matrix(matrix, seeds, epochs or None)
        configs = flatten(waves)
        log_dir = Path(VOL) / "results" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
        rec, rec_path = _live_record(matrix)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if rec is None:
            rec = B.new_record(matrix, budget_usd, TRAIN_CPU, TRAIN_MEM_GIB, driver_call_id=modal.current_function_call_id() or "",
                               note=f"seeds={seeds} gpu={gpu or 'default'} est={estimate_cost(configs, gpu or None)}")
            rec_path = LAUNCH_DIR / f"{matrix}_{stamp}.json"
            _save_record(rec, rec_path)
            log_path = log_dir / f"driver_{matrix}_{stamp}.log"
        else:
            log_path = log_dir / f"driver_{matrix}_{rec_path.stem.split('_', 1)[1]}.log"

        def log(line: str):
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")
            vol.commit()

        log(f"driver start matrix={matrix} seeds={seeds} runs={len(configs)} budget=${budget_usd} record={rec_path.name} "
            f"est={estimate_cost(configs, gpu or None)}" + (" (re-attached after restart)" if rec["runs"] else ""))
        try:
            _run_waves_budgeted(waves, configs, gpu, concurrency, log, rec, rec_path)
        except Exception as e:
            _cancel_all(rec, f"driver exception {e!r}", log); rec["status"] = B.HALTED; _save_record(rec, rec_path); raise
        return str(log_path)

    @app.local_entrypoint()
    def status(matrix: str = "", lines: int = 25):
        """From any machine with a Modal token: print the newest launch record (accrued cost, per-run
        elapsed/progress/projection) and the tail of its driver log. Touches no GPU."""
        vol.reload()
        recs = []
        try:
            listing = vol.listdir("results/launches")
        except Exception:
            listing = []
        for e in listing:
            name = e.path.split("/")[-1]
            if name.endswith(".json") and (not matrix or name.startswith(matrix + "_")):
                recs.append(e.path)
        if not recs:
            print("no launch records"); return
        latest = sorted(recs)[-1]
        rec = B.loads(b"".join(vol.read_file(latest)).decode())
        print(f"--- {latest}")
        print("\n".join(B.summary_lines(rec)))
        stamp = latest.split("/")[-1][len(rec["matrix"]) + 1:-5]
        try:
            text = b"".join(vol.read_file(f"results/logs/driver_{rec['matrix']}_{stamp}.log")).decode()
            print(f"--- driver log tail"); print("\n".join(text.splitlines()[-lines:]))
        except Exception as e:
            print(f"(driver log unavailable: {e!r})")

    @app.local_entrypoint()
    def halt(matrix: str = ""):
        """From any machine with a Modal token: cancel every live run of the newest live launch record
        (terminating containers) and cancel its driver. Idempotent."""
        vol.reload()
        found = False
        try:
            listing = [x.path for x in vol.listdir("results/launches")]
        except Exception:
            listing = []
        for e in sorted(listing, reverse=True):
            name = e.split("/")[-1]
            if matrix and not name.startswith(matrix + "_"):
                continue
            rec = B.loads(b"".join(vol.read_file(e)).decode())
            if rec["status"] != B.LIVE:
                continue
            found = True
            for n in B.live_runs(rec):
                cid = rec["runs"][n]["call_id"]
                try:
                    modal.FunctionCall.from_id(cid).cancel(terminate_containers=True); print(f"cancelled {n} {cid}")
                except Exception as ex:
                    print(f"cancel failed {n} {cid}: {ex!r}")
                B.finish_run(rec, n, "halted")
            if rec.get("driver_call_id"):
                try:
                    modal.FunctionCall.from_id(rec["driver_call_id"]).cancel(terminate_containers=True); print(f"cancelled driver {rec['driver_call_id']}")
                except Exception as ex:
                    print(f"driver cancel failed: {ex!r}")
            rec["status"] = B.HALTED; rec["events"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} halted from {os.uname().nodename}")
            with vol.batch_upload(force=True) as batch:
                batch.put_file(io.BytesIO(B.dumps(rec).encode()), e)
            print(f"halted {e}; accrued ${B.accrued_usd(rec):.2f}")
            break
        if not found:
            print("no live launch record; nothing to halt (check `modal app list` for stray apps)")

    @app.local_entrypoint()
    def sweep(matrix: str = "pilot", seeds: str = "0,1,2,3,4", gpu: str = "", epochs: int = 0,
              concurrency: int = 32, dry_run: bool = False, max_cost_usd: float = 0.0, driver: bool = False):
        seed_list = [int(s) for s in seeds.split(",") if s.strip()]
        print(f"mounted sprkd package: {SPRKD_PKG} ({package_version_info()})")
        if not (SPRKD_PKG / "hessian_utils.py").is_file():
            raise SystemExit("The mounted sprkd package predates the September 2026 fixes "
                             "(sprkd/hessian_utils.py missing). Check out main or neurips-fixes in the code repo.")
        waves = build_matrix(matrix, seed_list, epochs or None)
        configs = flatten(waves)
        est = estimate_cost(configs, gpu or None)
        print(f"matrix={matrix} waves={len(waves)} runs={len(configs)} est={est}")
        existing, _ = (None, None) if dry_run else _live_record(matrix)
        why = B.launch_refusal(est, max_cost_usd, driver, existing is not None, dry_run)
        if why:
            raise SystemExit(why)
        if dry_run:
            for wi, w in enumerate(waves):
                print(f"--- wave {wi + 1}: {len(w)} runs")
                for c in w:
                    print(f"  [{gpu_for(c, gpu or None)}] {c['run_name']}: {' '.join(cfg_to_argv(c, '<data>', '<out>', '<ckpt>'))}")
            return
        call = sweep_driver.spawn(matrix, seed_list, gpu, epochs, concurrency, max_cost_usd)
        print(f"driver spawned: call id {call.object_id}; budget ${max_cost_usd}; status: "
              f"`modal run neurips/bench/modal_app.py::status --matrix {matrix}`; stop: `modal run neurips/bench/modal_app.py::halt --matrix {matrix}`")
