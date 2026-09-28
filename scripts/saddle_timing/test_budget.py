"""Offline tests for the launch-record bookkeeping that bounds Modal spend."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import budget as B  # noqa: E402


def _rec(budget=10.0):
    rec = B.new_record("s1", budget, cpu_cores=4, mem_gib=16, now=1000.0)
    B.add_run(rec, "a", "fc-a", "A100", timeout_s=3600, now=1000.0)
    B.add_run(rec, "b", "fc-b", "A100", timeout_s=3600, now=1000.0)
    return rec


def test_all_in_rate_includes_cpu_and_memory():
    r = B.all_in_rate("A100", 8, 32)
    assert abs(r - (2.10 + 8 * 0.04716 + 32 * 0.007992)) < 1e-3
    assert B.all_in_rate("A100", 4, 16) < r


def test_accrued_cost_grows_with_time_and_stops_at_finish():
    rec = _rec()
    rate = rec["runs"]["a"]["rate_usd_h"]
    assert abs(B.accrued_usd(rec, now=1000.0 + 1800) - 2 * 0.5 * rate) < 1e-9
    B.finish_run(rec, "a", "ok", now=1000.0 + 1800)
    later = B.accrued_usd(rec, now=1000.0 + 3600)
    assert abs(later - (0.5 * rate + 1.0 * rate)) < 1e-9


def test_over_budget_fires_at_the_cap():
    rec = _rec(budget=1.0)
    rate = rec["runs"]["a"]["rate_usd_h"]
    seconds_to_cap = 1.0 / (2 * rate) * 3600
    assert not B.over_budget(rec, now=1000.0 + seconds_to_cap - 1)
    assert B.over_budget(rec, now=1000.0 + seconds_to_cap + 1)


def test_worst_case_is_bounded_by_timeouts():
    rec = _rec()
    rate = rec["runs"]["a"]["rate_usd_h"]
    assert abs(B.worst_case_usd(rec, now=1000.0) - 2 * rate) < 1e-9
    B.finish_run(rec, "a", "ok", now=1000.0 + 600)
    assert abs(B.worst_case_usd(rec, now=1000.0 + 600) - (rate + 600 / 3600 * rate)) < 1e-9


def test_heartbeat_stale():
    rec = _rec()
    assert not B.heartbeat_stale(rec, now=1000.0 + 600)
    assert B.heartbeat_stale(rec, now=1000.0 + 1500)
    rec["status"] = B.DONE
    assert not B.heartbeat_stale(rec, now=1000.0 + 99999)


def test_parse_progress_takes_last_line_and_ignores_noise():
    log = "\n".join([
        json.dumps({"seed": 0, "fraction": 0.5, "gn": 1.0}),
        json.dumps({"progress": 100, "of": 27000, "elapsed_s": 50.0}),
        "some warning text {not json}",
        json.dumps({"progress": 300, "of": 27000, "elapsed_s": 150.0, "note": "x"}),
    ])
    p = B.parse_progress(log)
    assert p["progress"] == 300 and p["of"] == 27000
    assert abs(B.projected_wall_s(p) - 13500.0) < 1e-6
    assert B.parse_progress("") is None


def test_should_abort_needs_evidence_then_fires():
    fast = {"progress": 300, "of": 27000, "elapsed_s": 150.0}      # projects 3.75 h
    assert not B.should_abort_run(fast, timeout_s=3600)               # too little elapsed evidence
    slow = {"progress": 300, "of": 27000, "elapsed_s": 900.0}      # projects 22.5 h
    assert B.should_abort_run(slow, timeout_s=3600)
    ok = {"progress": 13500, "of": 27000, "elapsed_s": 900.0}      # projects 0.5 h
    assert not B.should_abort_run(ok, timeout_s=3600)


def test_record_round_trips_and_summarises():
    rec = _rec()
    rec["runs"]["a"]["progress"] = {"progress": 10, "of": 100, "elapsed_s": 60.0}
    rec["events"].append("spawned a")
    back = B.loads(B.dumps(rec))
    assert back == rec
    lines = B.summary_lines(rec, now=1000.0 + 120)
    assert lines[0].startswith("launch s1 status=live budget=$10")
    assert any("projected=600s" in l for l in lines)


# --------------------------------------------------------------------------- driver loop

class FakeModal:
    """Fake clock + fake calls. Each run finishes after `durations[name]` seconds with rc 0,
    unless listed in `hang` (never finishes). Logs return `logs[name]` as a function of time."""

    def __init__(self, durations, hang=(), logs=None, poll_raises=()):
        self.t = 0.0; self.durations = durations; self.hang = set(hang); self.logs = logs or {}
        self.poll_raises = set(poll_raises)
        self.spawned = []; self.cancelled = []; self.started = {}; self.saved = 0; self.lines = []

    def now(self): return self.t
    def sleep(self, s): self.t += s
    def spawn(self, c):
        cid = f"fc-{c['run_name']}"; self.spawned.append(c["run_name"]); self.started[cid] = self.t
        return cid, "A100", c.get("timeout_s", 3600)
    def poll(self, cid):
        name = cid[3:]
        if name in self.poll_raises: raise RuntimeError("remote failure")
        if cid in self.cancelled or name in self.hang: return B.PENDING
        return {"returncode": 0, "wall_clock_s": self.t} if self.t - self.started[cid] >= self.durations[name] else B.PENDING
    def cancel(self, cid): self.cancelled.append(cid)
    def read_log(self, name):
        f = self.logs.get(name); return f(self.t - self.started[f"fc-{name}"]) if f else ""
    def save(self, rec): self.saved += 1
    def log(self, s): self.lines.append(s)

    def run(self, waves, rec, **kw):
        return B.run_budgeted(waves, rec, spawn=self.spawn, poll=self.poll, cancel=self.cancel, read_log=self.read_log,
                              save=self.save, log=self.log, now=self.now, sleep=self.sleep, **kw)


def _c(name, timeout_s=3600): return {"run_name": name, "timeout_s": timeout_s}


def test_driver_completes_waves_in_order_and_marks_done():
    fm = FakeModal({"a": 120, "b": 300, "c": 60})
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    st = fm.run([[_c("a"), _c("b")], [_c("c")]], rec)
    assert st == B.DONE and rec["status"] == B.DONE
    assert fm.spawned == ["a", "b", "c"] and fm.cancelled == []
    assert rec["runs"]["c"]["spawned_at"] >= rec["runs"]["b"]["finished_at"]   # wave 2 waits for wave 1
    assert all(r["status"] == "ok" for r in rec["runs"].values())


def test_driver_budget_stop_cancels_everything():
    fm = FakeModal({"a": 10**9, "b": 10**9})
    rec = B.new_record("m", 1.0, 4, 16, now=0.0)       # $1 budget, two A100s at ~$2.42/h each
    st = fm.run([[_c("a", 24 * 3600), _c("b", 24 * 3600)]], rec)
    assert st == B.BUDGET_STOP and rec["status"] == B.BUDGET_STOP
    assert sorted(fm.cancelled) == ["fc-a", "fc-b"]
    spent = B.accrued_usd(rec, fm.t)
    assert 1.0 <= spent < 1.0 + 2 * rec["runs"]["a"]["rate_usd_h"] * 30 / 3600 + 1e-9   # overshoot at most one poll interval


def test_driver_aborts_run_projecting_past_timeout():
    # S1-style run: progress says 20 of 27000 steps after 15 min -> projects ~140 days
    slow_log = lambda el: json.dumps({"progress": max(1, int(el / 45)), "of": 27000, "elapsed_s": el})
    fm = FakeModal({"slow": 10**9, "ok": 600}, logs={"slow": slow_log})
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    fm.run([[_c("slow", 7200), _c("ok", 7200)]], rec)
    assert rec["runs"]["slow"]["status"] == "aborted_projection"
    assert rec["runs"]["slow"]["finished_at"] <= 330           # caught within ~5.5 min, not 10 h
    assert fm.cancelled == ["fc-slow"] and rec["runs"]["ok"]["status"] == "ok"


def test_driver_aborts_silent_run_past_timeout():
    fm = FakeModal({}, hang=["mute"])                              # no progress lines at all, never returns
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    fm.run([[_c("mute", 900)]], rec, overrun_slack_s=600)
    assert rec["runs"]["mute"]["status"] == "aborted_overrun"
    assert 1500 <= rec["runs"]["mute"]["finished_at"] <= 1530


def test_driver_reattaches_after_restart_without_respawning():
    fm = FakeModal({"a": 600, "b": 600})
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    B.add_run(rec, "a", "fc-a", "A100", 3600, now=0.0); fm.started["fc-a"] = 0.0   # spawned before the "preemption"
    fm.run([[_c("a"), _c("b")]], rec)
    assert fm.spawned == ["b"]                                      # a was re-attached, not respawned
    assert rec["runs"]["a"]["status"] == "ok" and rec["runs"]["b"]["status"] == "ok"


def test_driver_records_remote_failure_and_continues():
    fm = FakeModal({"a": 60, "bad": 60}, poll_raises=["bad"])
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    assert fm.run([[_c("a"), _c("bad")]], rec) == B.DONE
    assert rec["runs"]["bad"]["status"].startswith("error:") and rec["runs"]["a"]["status"] == "ok"


def test_heartbeat_advances_while_running():
    fm = FakeModal({"a": 3000})
    rec = B.new_record("m", 100.0, 4, 16, now=0.0)
    fm.run([[_c("a")]], rec)
    assert rec["heartbeat_at"] >= 3000


# --------------------------------------------------------------------------- watchdog and launcher

def test_watchdog_reasons():
    rec = B.new_record("m", 1.0, 4, 16, now=0.0)
    B.add_run(rec, "a", "fc-a", "A100", 900, now=0.0)
    assert B.watchdog_reasons(rec, now=60.0) == []
    r = B.watchdog_reasons(rec, now=2000.0)                       # $1.34 accrued, heartbeat 2000 s old, 1100 s past a 900 s timeout
    assert any("budget" in x for x in r) and any("heartbeat" in x for x in r) and any("timeout" in x for x in r)
    rec["status"] = B.DONE
    assert B.watchdog_reasons(rec, now=99999.0) == []


def test_launch_refusal():
    est = {"worst_case_usd": 40.0, "usd": 20.0}
    assert B.launch_refusal(est, 50, driver=False, live_record_exists=False).startswith("refusing to run the wave loop")
    assert "without --max-cost-usd" in B.launch_refusal(est, 0, driver=True, live_record_exists=False)
    assert "worst case" in B.launch_refusal(est, 30, driver=True, live_record_exists=False)
    assert "live launch record" in B.launch_refusal(est, 50, driver=True, live_record_exists=True)
    assert B.launch_refusal(est, 50, driver=True, live_record_exists=False) is None
    assert B.launch_refusal(est, 0, driver=False, live_record_exists=True, dry_run=True) is None
