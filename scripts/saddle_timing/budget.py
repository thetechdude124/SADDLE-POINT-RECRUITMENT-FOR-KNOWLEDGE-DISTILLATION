"""Pure bookkeeping for Modal launches: launch records, accrued cost, progress projection.

Nothing here imports modal, so it is unit-testable offline. The cloud driver, the deployed
watchdog, and the `status` / `halt` entrypoints all read and write the same launch record,
a JSON file on the volume at ``results/launches/<matrix>_<timestamp>.json``.

Why this exists: on 2026-09-28 stage S1 ran for ten hours on six A100s with no runtime
budget, no progress signal, and a driver that respawned everything after a preemption.
See ``neurips/10_s1_incident.md``.
"""

from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

LIVE = "live"
DONE = "done"
BUDGET_STOP = "budget_stop"
HALTED = "halted"
STALE_STOP = "stale_stop"

# Modal list prices read 2026-09-28 (modal.com/pricing). GPU per hour; CPU per core-hour;
# memory per GiB-hour. A container bills GPU + requested CPU + requested memory.
GPU_USD_PER_HOUR = {"H100": 3.95, "A100-80GB": 2.50, "A100": 2.10, "A10G": 1.10, "L4": 0.80, "T4": 0.59}
CPU_USD_PER_CORE_HOUR = 0.0000131 * 3600
MEM_USD_PER_GIB_HOUR = 0.00000222 * 3600


def all_in_rate(gpu: str, cpu_cores: float, mem_gib: float) -> float:
    """USD per hour for one container with this GPU and these CPU/memory reservations."""
    return GPU_USD_PER_HOUR.get(gpu, 2.10) + cpu_cores * CPU_USD_PER_CORE_HOUR + mem_gib * MEM_USD_PER_GIB_HOUR


def new_record(matrix: str, budget_usd: float, cpu_cores: float, mem_gib: float, now: Optional[float] = None,
               driver_call_id: str = "", app_id: str = "", note: str = "") -> dict:
    now = time.time() if now is None else now
    return {"matrix": matrix, "status": LIVE, "budget_usd": float(budget_usd), "cpu_cores": cpu_cores, "mem_gib": mem_gib,
            "created_at": now, "heartbeat_at": now, "driver_call_id": driver_call_id, "app_id": app_id, "note": note,
            "runs": {}, "events": []}


def add_run(rec: dict, run_name: str, call_id: str, gpu: str, timeout_s: int, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    rec["runs"][run_name] = {"call_id": call_id, "gpu": gpu, "timeout_s": int(timeout_s), "spawned_at": now,
                             "finished_at": None, "status": "running", "rate_usd_h": all_in_rate(gpu, rec["cpu_cores"], rec["mem_gib"]),
                             "progress": None}


def finish_run(rec: dict, run_name: str, status: str, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    r = rec["runs"][run_name]
    if r["finished_at"] is None:
        r["finished_at"] = now
    r["status"] = status


def live_runs(rec: dict) -> List[str]:
    return [n for n, r in rec["runs"].items() if r["finished_at"] is None]


def run_elapsed_s(r: dict, now: float) -> float:
    end = r["finished_at"] if r["finished_at"] is not None else now
    return max(0.0, end - r["spawned_at"])


def accrued_usd(rec: dict, now: Optional[float] = None) -> float:
    """Cost so far: every run's elapsed time at its all-in rate. Slightly pessimistic (a
    queued run is charged from spawn, not from container start)."""
    now = time.time() if now is None else now
    return sum(run_elapsed_s(r, now) / 3600.0 * r["rate_usd_h"] for r in rec["runs"].values())


def worst_case_usd(rec: dict, now: Optional[float] = None) -> float:
    """Accrued cost plus every live run running to its hard timeout."""
    now = time.time() if now is None else now
    total = 0.0
    for r in rec["runs"].values():
        if r["finished_at"] is None:
            total += r["timeout_s"] / 3600.0 * r["rate_usd_h"]
        else:
            total += run_elapsed_s(r, now) / 3600.0 * r["rate_usd_h"]
    return total


def burn_usd_per_s(rec: dict) -> float:
    """Current spend rate of all live runs."""
    return sum(r["rate_usd_h"] for r in rec["runs"].values() if r["finished_at"] is None) / 3600.0


def over_budget(rec: dict, now: Optional[float] = None, lookahead_s: float = 0.0) -> bool:
    """True when accrued cost, plus what live runs will burn in the next ``lookahead_s``, reaches
    the budget. The driver passes one poll interval (plus margin) so it stops before the cap
    instead of up to one interval after it (measured overshoot on Modal: $0.06 on a $0.03 budget)."""
    return accrued_usd(rec, now) + burn_usd_per_s(rec) * lookahead_s >= rec["budget_usd"]


def heartbeat_stale(rec: dict, now: Optional[float] = None, max_age_s: float = 20 * 60) -> bool:
    now = time.time() if now is None else now
    return rec["status"] == LIVE and (now - rec["heartbeat_at"]) > max_age_s


_PROGRESS = re.compile(r'\{[^{}]*"progress"[^{}]*\}')


def parse_progress(log_text: str) -> Optional[dict]:
    """Last progress line in a run log. Runs print ``{"progress": step, "of": total, "elapsed_s": t}``."""
    last = None
    for m in _PROGRESS.finditer(log_text):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if {"progress", "of", "elapsed_s"} <= set(d):
            last = d
    return last


def projected_wall_s(progress: Optional[dict]) -> Optional[float]:
    if not progress or not progress.get("progress"):
        return None
    return progress["elapsed_s"] * progress["of"] / progress["progress"]


def should_abort_run(progress: Optional[dict], timeout_s: int, min_elapsed_s: float = 300.0, slack: float = 1.0) -> bool:
    """True when a run's own progress line projects a wall-clock beyond its hard timeout.
    Requires at least ``min_elapsed_s`` of evidence so a slow first step cannot trigger it."""
    if not progress or progress["elapsed_s"] < min_elapsed_s:
        return False
    proj = projected_wall_s(progress)
    return proj is not None and proj > slack * timeout_s


def summary_lines(rec: dict, now: Optional[float] = None) -> List[str]:
    now = time.time() if now is None else now
    out = [f"launch {rec['matrix']} status={rec['status']} budget=${rec['budget_usd']:.2f} accrued=${accrued_usd(rec, now):.2f} "
           f"worst_case=${worst_case_usd(rec, now):.2f} heartbeat_age={int(now - rec['heartbeat_at'])}s runs={len(rec['runs'])} live={len(live_runs(rec))}"]
    for n, r in rec["runs"].items():
        p = r.get("progress"); proj = projected_wall_s(p)
        out.append(f"  {n:<28} {r['gpu']:<5} {r['status']:<10} elapsed={int(run_elapsed_s(r, now))}s timeout={r['timeout_s']}s "
                   f"progress={p['progress'] if p else '-'}/{p['of'] if p else '-'} projected={int(proj) if proj else '-'}s call={r['call_id']}")
    for e in rec["events"][-5:]:
        out.append(f"  event: {e}")
    return out


def dumps(rec: dict) -> str:
    return json.dumps(rec, indent=1)


def loads(text: str) -> dict:
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Driver loop and watchdog decisions, with every Modal interaction injected so the
# control flow is unit-tested offline (tests/test_budget.py) before any paid launch.
# --------------------------------------------------------------------------- #

PENDING = object()  # poll() returns this while a call is still running


def run_budgeted(waves, rec, *, spawn, poll, cancel, read_log, save, log, now, sleep,
                 poll_s: float = 30.0, overrun_slack_s: float = 600.0):
    """Run waves of configs under the budget in ``rec``.

    spawn(cfg) -> (call_id, gpu, timeout_s); poll(call_id) -> result dict, PENDING, or raises;
    cancel(call_id) terminates the call's container; read_log(run_name) -> str;
    save(rec) persists the record; now() -> float; sleep(s).

    Runs already in ``rec["runs"]`` are re-attached, never respawned (driver restarted after
    a preemption). Stops everything when accrued cost reaches the budget. Aborts a run whose
    progress projects past its timeout, or that is still alive well past its timeout.
    Returns the final status.
    """
    for wi, w in enumerate(waves):
        pending = [c for c in w if c["run_name"] not in rec["runs"]]
        log(f"=== wave {wi + 1}/{len(waves)}: {len(w)} runs ({len(w) - len(pending)} already recorded)")
        for c in pending:
            call_id, gpu, tmo = spawn(c)
            add_run(rec, c["run_name"], call_id, gpu, tmo, now())
            log(f"spawned {c['run_name']} on {gpu} timeout={tmo}s call={call_id}")
        save(rec)
        names = [c["run_name"] for c in w]
        while any(rec["runs"][n]["finished_at"] is None for n in names):
            sleep(poll_s)
            t = now()
            for n in names:
                r = rec["runs"][n]
                if r["finished_at"] is not None:
                    continue
                try:
                    res = poll(r["call_id"])
                except Exception as e:
                    finish_run(rec, n, f"error:{type(e).__name__}", t); log(f"{n} ended with {e!r}"); continue
                if res is not PENDING:
                    ok = isinstance(res, dict) and res.get("returncode") == 0 and not res.get("error")
                    finish_run(rec, n, "ok" if ok else "failed", t)
                    log(f"{n} {'ok' if ok else 'FAILED'} " + (str({k: res.get(k) for k in ('returncode', 'wall_clock_s')}) if isinstance(res, dict) else repr(res)))
                    continue
                prog = parse_progress(read_log(n))
                if prog is not None and prog["elapsed_s"] > run_elapsed_s(r, t) + 120:
                    prog = None  # claims more elapsed time than this run has existed: a stale log from an earlier launch
                r["progress"] = prog
                if should_abort_run(r["progress"], r["timeout_s"]):
                    cancel(r["call_id"]); finish_run(rec, n, "aborted_projection", t)
                    log(f"ABORT {n}: progress projects {int(projected_wall_s(r['progress']))}s against a {r['timeout_s']}s timeout")
                    continue
                if run_elapsed_s(r, t) > r["timeout_s"] + overrun_slack_s:
                    cancel(r["call_id"]); finish_run(rec, n, "aborted_overrun", t)
                    log(f"ABORT {n}: alive {int(run_elapsed_s(r, t))}s past a {r['timeout_s']}s timeout")
            rec["heartbeat_at"] = t
            if over_budget(rec, t, lookahead_s=1.5 * poll_s + 15):
                for n in live_runs(rec):
                    cancel(rec["runs"][n]["call_id"]); finish_run(rec, n, "cancelled", t)
                rec["status"] = BUDGET_STOP
                rec["events"].append(f"budget stop at accrued ${accrued_usd(rec, t):.2f}")
                save(rec); log(f"BUDGET STOP: accrued ${accrued_usd(rec, t):.2f} >= ${rec['budget_usd']:.2f}")
                return BUDGET_STOP
            save(rec)
    rec["status"] = DONE; save(rec)
    log(f"done. accrued ${accrued_usd(rec, now()):.2f}")
    return DONE


def watchdog_reasons(rec: dict, now: float, stale_s: float = 20 * 60, overrun_slack_s: float = 600.0) -> List[str]:
    """Why the deployed watchdog should kill this launch (empty list = leave it alone)."""
    if rec["status"] != LIVE:
        return []
    out = []
    if over_budget(rec, now):
        out.append(f"accrued ${accrued_usd(rec, now):.2f} >= budget ${rec['budget_usd']:.2f}")
    if heartbeat_stale(rec, now, stale_s):
        out.append(f"driver heartbeat {int(now - rec['heartbeat_at'])}s old")
    over = [n for n in live_runs(rec) if run_elapsed_s(rec["runs"][n], now) > rec["runs"][n]["timeout_s"] + overrun_slack_s]
    if over:
        out.append(f"runs past their timeout: {over}")
    return out


def launch_refusal(est: dict, max_cost_usd: float, driver: bool, live_record_exists: bool, dry_run: bool = False,
                   allow_unmeasured: bool = False) -> Optional[str]:
    """Reason the launcher must refuse, or None. Pure so it is tested offline."""
    if dry_run:
        return None
    if est.get("unmeasured") and not allow_unmeasured:
        return (f"refusing: no measured step time on the target GPU for {est['unmeasured'][:5]}"
                f"{' ...' if len(est['unmeasured']) > 5 else ''}; run a timing probe first (or --allow-unmeasured)")
    if not driver:
        return "refusing to run the wave loop from this machine; launch with --driver --detach"
    if not max_cost_usd:
        return "refusing to launch without --max-cost-usd"
    if est["worst_case_usd"] > max_cost_usd:
        return f"refusing: worst case ${est['worst_case_usd']} exceeds --max-cost-usd {max_cost_usd}"
    if live_record_exists:
        return "refusing: a live launch record for this matrix exists; halt it first"
    return None
