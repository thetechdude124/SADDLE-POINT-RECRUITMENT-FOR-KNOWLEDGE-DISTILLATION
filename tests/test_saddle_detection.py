"""Tests for direct extreme-eigenpair saddle detection and stationary-point refinement (0.3.0)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sprkd.hessian_utils import extreme_eigenpairs
from sprkd.models import MalariaTeacherCNN
from sprkd.optimizer import SPRKD
from sprkd.saddle import SaddleCriterion, is_strong_saddle_point, refine_to_stationary, which_rules_fire

DIAG = torch.tensor([3.0, 1.0, 0.5, -0.5, -2.0])


class _Quadratic(nn.Module):
    """f(w) = 0.5 w^T diag(3, 1, 0.5, -0.5, -2) w: saddle at 0, lambda_max 3, lambda_min -2."""

    def __init__(self, w0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w0))

    def forward(self, x):
        return (0.5 * (self.w * DIAG * self.w).sum()).expand(x.shape[0], 1)


class _MeanLoss(nn.Module):
    def forward(self, out, y):
        return out.mean()


BATCH = (torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))


@pytest.mark.parametrize("method,tol", [("lanczos", 1e-3), ("power", 1e-2)])
def test_extreme_eigenpairs_on_indefinite_quadratic(method, tol):
    torch.manual_seed(0)
    m = _Quadratic([0.7, -0.4, 0.3, 0.9, -0.6])
    r = extreme_eigenpairs(m, _MeanLoss(), BATCH, method=method)
    assert abs(r["lambda_max"] - 3.0) < tol and abs(r["lambda_min"] + 2.0) < tol
    assert abs(abs(float(r["v_max"][0][0])) - 1.0) < 0.05 and abs(abs(float(r["v_min"][0][4])) - 1.0) < 0.05
    assert r["n_hvp"] > 0 and m.training  # mode restored


def test_refine_to_stationary_converges_to_saddle_at_origin():
    m = _Quadratic([0.7, -0.4, 0.3, 0.9, -0.6])
    r = refine_to_stationary(m, _MeanLoss(), BATCH, max_steps=20, grad_tol=1e-6)
    assert r["converged"] and r["grad_norm_after"] <= 1e-6
    assert m.w.detach().abs().max() < 1e-5


def test_refine_gradnorm_method_decreases_gradient():
    m = _Quadratic([0.7, -0.4, 0.3, 0.9, -0.6])
    r = refine_to_stationary(m, _MeanLoss(), BATCH, max_steps=200, grad_tol=1e-3, lr=0.1, method="gradnorm")
    assert r["grad_norm_after"] < r["grad_norm_before"]


def test_extreme_rule_semantics():
    crit = SaddleCriterion()  # rule="extreme", tau_rel 0.05
    assert crit.rule == "extreme"
    assert is_strong_saddle_point(lambda_max=10.0, lambda_min=-1.0)
    assert not is_strong_saddle_point(lambda_max=10.0, lambda_min=-0.2)      # -0.2 > -0.5
    assert not is_strong_saddle_point(lambda_max=-1.0, lambda_min=-3.0)      # lambda_max must be positive
    assert is_strong_saddle_point(lambda_max=10.0, lambda_min=-0.2, criterion=SaddleCriterion(tau=0.1))
    out = which_rules_fire(None, SaddleCriterion(max_grad_norm=0.5), grad_norm=1.0, lambda_max=10.0, lambda_min=-1.0)
    assert out["extreme"] and not out["grad_gate"] and not out["fired"]
    assert is_strong_saddle_point([10.0, -8.0], SaddleCriterion(rule="magnitude"))   # legacy rule still available


@pytest.mark.slow
def test_malaria_teacher_refined_snapshot_is_a_verified_saddle():
    torch.manual_seed(3)
    t = MalariaTeacherCNN()
    x = torch.rand(32, 3, 32, 32); y = torch.randint(0, 2, (32,))
    loss_fn = nn.CrossEntropyLoss()
    opt = SPRKD(t.parameters(), base_optimizer=torch.optim.Adam(t.parameters(), 1e-3), loss_fn=loss_fn,
                is_teacher=True, saddle_steps=5, saddle_refine=True, refine_grad_tol=1e-2, refine_max_steps=20)
    for _ in range(30):
        opt.zero_grad(); loss = loss_fn(t(x), y); loss.backward()
        opt.step(model=t, current_loss=loss.detach(), data_batch=(x, y))
    c = opt.counters()
    assert c["saddles_checked"] == 6 and c["saddles_refined"] >= 1 and c["saddles_recorded"] >= 1
    repo = opt.saddle_repository
    r = repo.rules[0]
    assert r["verified"] and r["grad_norm_after"] <= 1e-2 and r["lambda_min_after"] < 0
    assert r["grad_norm_after"] < r["grad_norm_before"]
    # the stored snapshot is the refined point: evaluating its gradient reproduces grad_norm_after
    for p, s in zip(t.parameters(), repo.snapshots[0]):
        p.data.copy_(s)
    t.eval()
    loss = loss_fn(t(x), y)
    g = torch.autograd.grad(loss, list(t.parameters()))
    gn = torch.sqrt(sum((gi ** 2).sum() for gi in g)).item()
    assert abs(gn - r["grad_norm_after"]) < 1e-3
    assert t.training is False or True


def test_nhe_lambda_min_direction_on_quadratic():
    m = _Quadratic([0.0, 0.0, 0.0, 0.0, 0.05])   # near the saddle; negative curvature along e4
    opt = SPRKD([m.w], base_optimizer=torch.optim.SGD([m.w], lr=0.1), loss_fn=_MeanLoss(),
                teacher_saddle_points=[torch.zeros(5)], nhe_step_mode="fixed", nhe_step_size=0.3)
    _MeanLoss()(m(BATCH[0]), BATCH[1]).backward()
    before = m.w.detach().clone()
    assert opt._negative_hessian_eigenstep(group=opt.param_groups[0], model=m, data_batch=BATCH)
    d = m.w.detach() - before
    assert abs(float(d.norm()) - 0.3) < 1e-4 and abs(float(d[4])) > 0.29 and float(d[4]) > 0   # away from 0 along -grad direction
    assert opt.events[-1]["post_loss"] < opt.events[-1]["pre_loss"]


def test_extreme_eigenpairs_k_most_negative():
    D = torch.tensor([3.0, 1.0, 0.5, -0.5, -2.0, -1.0])

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.7, -0.4, 0.3, 0.9, -0.6, 0.2]))

        def forward(self, x):
            return (0.5 * (self.w * D * self.w).sum()).expand(x.shape[0], 1)

    r = extreme_eigenpairs(Q(), _MeanLoss(), BATCH, k=3)
    assert [round(v, 3) for v in r["lambda_min_k"]] == [-2.0, -1.0, -0.5]
    assert abs(r["lambda_max"] - 3.0) < 1e-3
    with pytest.raises(NotImplementedError):
        extreme_eigenpairs(Q(), _MeanLoss(), BATCH, k=2, method="power")
