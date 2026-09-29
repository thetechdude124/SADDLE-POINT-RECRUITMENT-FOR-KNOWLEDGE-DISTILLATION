"""Turn the gate_probe runs' measured epoch times into a cost for the full gate.

Reads results/runs/<probe run>/epochs.csv and summary.json from the Modal volume (via the
modal CLI, no GPU), takes the second epoch's wall time (the first includes data loading and
cudnn warm-up), and prices 240-epoch runs at Modal list prices incl. CPU and memory.

Usage: conda run -n sprkd python neurips/bench/gate_cost_from_probe.py
"""

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget as B  # noqa: E402

VOL = "sprkd-bench"
RUNS = {"scratch": "gprobe_{g}_resnet8x4_scratch", "init": "gprobe_{g}_resnet8x4_lastinit_f050", "kd": "gprobe_{g}_resnet8x4_kdweak"}
EPOCHS = 240
SEEDS = 3
ARMS_PER_SEED = {"scratch": 1, "init": 3, "kd": 1}      # the gate: scratch, init at 20/50/100%, KD-weak
TIMEOUT_FACTOR = 2.0
STARTUP_H = 3 / 60


def vol_get(path):
    r = subprocess.run(["modal", "volume", "get", VOL, path, "-"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _latest_launch_id():
    r = subprocess.run(["modal", "volume", "ls", VOL, "results/launches"], capture_output=True, text=True)
    ids = sorted(l.strip().split("/")[-1][:-5] for l in r.stdout.splitlines() if "gate_probe_" in l and l.strip().endswith(".json"))
    return ids[-1] if ids else None


def epoch_seconds(run, launch_id):
    """Second epoch's wall time from the run's per-launch log (JSON epoch rows; the first epoch
    includes data loading and cudnn warm-up). The log is used rather than epochs.csv because the
    probe ran before the CSV column fix."""
    text = vol_get(f"results/logs/{launch_id}__{run}.log")
    if not text:
        return None, "no log"
    rows = []
    for line in text.splitlines():
        if line.startswith("{") and '"epoch"' in line and '"progress"' not in line:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("phase", "student") == "student" and d.get("epoch", 0) >= 1 and "elapsed_s" in d:
                rows.append(d)
    if len(rows) < 2:
        return None, f"only {len(rows)} student epochs"
    return rows[-1]["elapsed_s"] - rows[-2]["elapsed_s"], f"epochs={len(rows)} test_top1={rows[-1].get('test_top1')}"


def main():
    out = {}
    launch_id = _latest_launch_id()
    print("probe launch:", launch_id)
    for g in ("a100", "t4"):
        gpu = "A100" if g == "a100" else "T4"
        rate = B.all_in_rate(gpu, 4, 16)
        per = {}
        for arm, pat in RUNS.items():
            s, note = epoch_seconds(pat.format(g=g), launch_id)
            per[arm] = s
            print(f"{gpu:5} {arm:8} epoch_s={s} ({note})")
        if any(v is None for v in per.values()):
            print(f"{gpu}: incomplete, not pricing"); continue
        hours = {arm: STARTUP_H + EPOCHS * per[arm] / 3600 for arm in per}
        est = sum(SEEDS * ARMS_PER_SEED[a] * hours[a] * rate for a in hours)
        worst = sum(SEEDS * ARMS_PER_SEED[a] * max(0.25, TIMEOUT_FACTOR * hours[a]) * rate for a in hours)
        wall = max(hours.values())
        out[gpu] = {"epoch_s": per, "run_hours": {a: round(h, 2) for a, h in hours.items()}, "est_usd": round(est, 2),
                    "worst_usd": round(worst, 2), "wall_h": round(wall, 2), "rate_usd_h": round(rate, 3)}
        print(f"{gpu}: 15 student runs x 240 epochs  est ${est:.2f}  worst ${worst:.2f}  wall-clock ~{wall:.1f} h (parallel)")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
