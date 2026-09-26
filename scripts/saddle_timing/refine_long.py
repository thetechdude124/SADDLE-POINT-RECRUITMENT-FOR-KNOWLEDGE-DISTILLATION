"""Does refinement of a mid-training checkpoint ever converge with a much larger budget?"""
import json, sys, time, torch, torch.nn as nn
sys.path.insert(0, '.')
from saddle_timing_study import Data, evaluate, flat
from sprkd.models import MalariaTeacherCNN
from sprkd.saddle import refine_to_stationary
from sprkd.hessian_utils import extreme_eigenpairs
data = Data(torch.device("cpu"), 256)
out = []
for method, steps, lr in (("gn", 400, 1e-2), ("gradnorm", 400, 1e-3), ("gradnorm", 400, 1e-4)):
    t = MalariaTeacherCNN(); t.load_state_dict(torch.load("results/saddle_timing/teacher_s0/ck_035.pt", map_location="cpu"))
    th0 = flat(t); t0 = time.time()
    r = refine_to_stationary(t, nn.CrossEntropyLoss(), data.probe, max_steps=steps, grad_tol=1e-2, lr=lr, method=method, solver_iters=60)
    e = extreme_eigenpairs(t, nn.CrossEntropyLoss(), data.probe, k=3, tol=1e-2, max_iter=20)
    va, _ = evaluate(t, data.x_va, data.y_va)
    rec = {"method": method, "max_steps": steps, "lr": lr, **{k: v for k, v in r.items()}, "lambda_max_after": e["lambda_max"], "lambda_min_k_after": e["lambda_min_k"],
           "val_acc_after": va["acc"], "dist_rel": float((flat(t) - th0).norm() / th0.norm()), "wall_s": round(time.time() - t0, 1)}
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open("results/saddle_timing/refine_long_s0_035.json", "w"), indent=2)
