"""Recompute the straight-line loss profile between the two teachers of each NEB pair (the
in-run version multiplied an MPS tensor by a CPU 0-dim tensor and got the A endpoint back)."""
import json, sys, torch
sys.path.insert(0, ".")
from saddle_timing_study import Data, flat
from neb_ridge_saddle import FlatLoss, get_teacher, OUT
from common import get_device
import argparse
p = argparse.ArgumentParser(); p.add_argument("--pairs", default="0:1"); a = p.parse_args()
device = get_device(); data = Data(device, 256)
out = json.load(open(OUT / "linear_paths.json")) if (OUT / "linear_paths.json").is_file() else {}
class A: epochs = 40
for pair in a.pairs.split(","):
    sa, sb = [int(v) for v in pair.split(":")]
    tA, _ = get_teacher(sa, A, data, device); tB, _ = get_teacher(sb, A, data, device)
    fl = FlatLoss(tA, data.x_tr[:8], data.y_tr[:8]); va, vb = flat(tA).to(device), flat(tB).to(device)
    tr = [fl.loss_batched(va + float(t) * (vb - va), data.x_tr, data.y_tr) for t in torch.linspace(0, 1, 11)]
    vl = [fl.loss_batched(va + float(t) * (vb - va), data.x_va, data.y_va) for t in torch.linspace(0, 1, 11)]
    out[f"pair_{sa}_{sb}"] = {"train": tr, "val": vl, "barrier_train": max(tr) - 0.5 * (tr[0] + tr[-1]), "barrier_val": max(vl) - 0.5 * (vl[0] + vl[-1])}
    print(json.dumps({pair: {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in out[f"pair_{sa}_{sb}"].items()}}))
json.dump(out, open(OUT / "linear_paths.json", "w"), indent=2)
