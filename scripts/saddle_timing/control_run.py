"""Control runner for verifying the Modal budget machinery for cents, no GPU work.

Prints the same heartbeat lines a real run prints, then either finishes or deliberately
overruns so the driver's projection abort / budget stop / halt can be exercised.
"""
import argparse, json, time
p = argparse.ArgumentParser()
p.add_argument("--out-root", default="."); p.add_argument("--data-root", default="."); p.add_argument("--run-name", required=True)
p.add_argument("--total-steps", type=int, default=60); p.add_argument("--step-s", type=float, default=1.0)
p.add_argument("--report-of", type=int, default=0, help="claim this many total steps in progress lines (to fake a slow projection)")
a, _ = p.parse_known_args()
t0 = time.time(); of = a.report_of or a.total_steps
for i in range(1, a.total_steps + 1):
    time.sleep(a.step_s)
    if i % 5 == 0 or i == a.total_steps:
        print(json.dumps({"progress": i, "of": of, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
print(json.dumps({"done": True, "run_name": a.run_name, "wall_s": round(time.time() - t0, 1)}), flush=True)
