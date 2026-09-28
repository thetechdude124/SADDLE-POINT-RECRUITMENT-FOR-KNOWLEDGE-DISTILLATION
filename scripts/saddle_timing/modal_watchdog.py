"""Deployed watchdog: kills any sprkd-bench launch that is over budget or whose driver died.

Runs every 5 minutes inside Modal (no laptop involved) once deployed with

    modal deploy neurips/bench/modal_watchdog.py

It reads every launch record under results/launches/ on the shared volume and, for each
record still marked live:

* if accrued cost >= budget, or
* if the driver heartbeat is older than STALE_S (the driver was preempted, crashed, or was
  never restarted), or
* if any run has exceeded its hard timeout by more than 10 minutes,

it cancels every live run with container termination, cancels the driver call, and marks the
record. It costs a few CPU-seconds per tick. This is the backstop behind the driver's own
budget loop; both exist because on 2026-09-28 a driver-only design lost $140
(see neurips/10_s1_incident.md).

Manual check from any machine:  modal run neurips/bench/modal_watchdog.py::tick
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget as B  # noqa: E402

VOL_NAME = "sprkd-bench"
VOL = "/vol"
STALE_S = 20 * 60
OVERRUN_SLACK_S = 10 * 60

vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.10").add_local_file(
    str(Path(__file__).resolve().parent / "budget.py"), remote_path="/root/budget.py")
app = modal.App("sprkd-watchdog")


def _sweep_once(log) -> int:
    vol.reload()
    launch_dir = Path(VOL) / "results" / "launches"
    launch_dir.mkdir(parents=True, exist_ok=True)
    now = time.time(); acted = 0
    for path in sorted(launch_dir.glob("*.json")):
        rec = B.loads(path.read_text())
        if rec["status"] != B.LIVE:
            continue
        reasons = B.watchdog_reasons(rec, now, rec.get("stale_s", STALE_S), OVERRUN_SLACK_S)
        if not reasons:
            log(f"{path.name}: live, accrued ${B.accrued_usd(rec, now):.2f} of ${rec['budget_usd']:.0f}, {len(B.live_runs(rec))} runs, heartbeat {int(now - rec['heartbeat_at'])}s ago")
            continue
        reason = "WATCHDOG STOP: " + "; ".join(reasons)
        log(f"{path.name}: {reason}")
        for n in B.live_runs(rec):
            cid = rec["runs"][n]["call_id"]
            try:
                modal.FunctionCall.from_id(cid).cancel(terminate_containers=True); log(f"  cancelled {n} {cid}")
            except Exception as e:
                log(f"  cancel failed {n} {cid}: {e!r}")
            B.finish_run(rec, n, "watchdog_cancelled", now)
        if rec.get("driver_call_id"):
            try:
                modal.FunctionCall.from_id(rec["driver_call_id"]).cancel(terminate_containers=True); log(f"  cancelled driver {rec['driver_call_id']}")
            except Exception as e:
                log(f"  driver cancel failed: {e!r}")
        rec["status"] = B.BUDGET_STOP if reasons and reasons[0].startswith("accrued") else B.STALE_STOP
        rec["events"].append(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {reason}")
        path.write_text(B.dumps(rec)); vol.commit(); acted += 1
    return acted


@app.function(image=image, volumes={VOL: vol}, schedule=modal.Period(minutes=5), timeout=300, cpu=0.25, memory=512)
def tick():
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)
    acted = _sweep_once(log)
    log_path = Path(VOL) / "results" / "logs" / "watchdog.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} acted={acted}\n" + "\n".join(lines) + "\n")
    vol.commit()
    return acted


@app.local_entrypoint()
def check():
    """Run one watchdog sweep now from this machine and print what it saw."""
    print(tick.remote())
