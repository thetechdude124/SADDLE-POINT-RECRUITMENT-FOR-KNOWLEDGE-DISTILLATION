"""Regression tests for the September 2026 corrections (see README, "Errata and
corrections", and neurips/04_paper_vs_code_reconciliation.md). One test per item."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sprkd.hessian_utils import preserve_model_state, top_eigenpairs
from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.optimizer import SPRKD
from sprkd.saddle import (
    SaddleCriterion,
    SaddlePointRepository,
    aggregate_asr,
    is_strong_saddle_point,
    which_rules_fire,
)
from sprkd.tli import inject_state_list, transfer_via_graph
from sprkd.training import kd_loss, train_response_kd


# (1) logits ------------------------------------------------------------------

def test_fix1_models_return_logits_and_losses_work(tiny_batch):
    x, y = tiny_batch
    for m in (MalariaTeacherCNN(), MalariaStudentCNN()):
        assert not any(isinstance(mod, nn.Softmax) for mod in m.modules())
        out = m(x)
        # logits are unconstrained: a softmax of them is a distribution, they are not
        assert torch.allclose(torch.softmax(out, 1).sum(1), torch.ones(len(y)), atol=1e-5)
        loss = nn.functional.cross_entropy(out, y)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None for p in m.parameters())
        assert (out.argmax(1) == torch.softmax(out, 1).argmax(1)).all()


def test_fix_kd_loss_is_standard_hinton(tiny_batch):
    _, y = tiny_batch
    s = torch.randn(8, 2, requires_grad=True)
    t = torch.randn(8, 2)
    # alpha = 1 -> pure cross-entropy
    assert torch.allclose(kd_loss(s, t, y, alpha=1.0), nn.functional.cross_entropy(s, y))
    # alpha = 0 and identical logits -> zero
    assert kd_loss(s, s.detach(), y, alpha=0.0, temperature=4.0).abs() < 1e-5
    # KL term is T^2-scaled KL(softmax(t/T) || softmax(s/T))
    T = 4.0
    kl = nn.functional.kl_div(torch.log_softmax(s / T, 1), torch.softmax(t / T, 1), reduction="batchmean") * T * T
    assert torch.allclose(kd_loss(s, t, y, alpha=0.0, temperature=T), kl)
    with pytest.raises(ValueError):
        kd_loss(s, t, y, alpha=1.5)


def test_fix_kd_baseline_trains_on_labels(tiny_loader, cpu_loss):
    student, teacher = MalariaStudentCNN(), MalariaTeacherCNN()
    history = train_response_kd(student, teacher, tiny_loader, tiny_loader, n_epochs=1, alpha=0.5, temperature=4.0, progress=False)
    assert len(history.train_losses) == len(tiny_loader)
    assert all(torch.isfinite(torch.tensor(v)) for v in history.train_losses)


# (2) PyHessian side effects ----------------------------------------------------

def test_fix2_hessian_calls_preserve_grads_and_train_mode(student_model, tiny_batch, cpu_loss):
    pytest.importorskip("pyhessian")
    x, y = tiny_batch
    student_model.train()
    loss = cpu_loss(student_model(x), y)
    loss.backward()
    grads = [p.grad.clone() for p in student_model.parameters()]
    eigs, vecs = top_eigenpairs(student_model, cpu_loss, (x, y), top_n=1)
    assert len(eigs) == 1
    assert student_model.training, "train mode must be restored after a Hessian call"
    for p, g in zip(student_model.parameters(), grads):
        assert p.grad is not None and torch.allclose(p.grad, g)


def test_fix2_preserve_model_state_context(student_model):
    student_model.eval()
    for p in student_model.parameters():
        p.grad = torch.ones_like(p)
    with preserve_model_state(student_model):
        student_model.train()
        student_model.zero_grad(set_to_none=True)
    assert not student_model.training
    assert all(p.grad is not None and torch.all(p.grad == 1) for p in student_model.parameters())


# (3) NHE ------------------------------------------------------------------------

class _QuadraticSaddle(nn.Module):
    """f(w) = 0.5 * (w0^2 - w1^2): a saddle at 0 with negative curvature along e1."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.3, 0.05]))

    def forward(self, x):
        # returns a (batch, 1) "logit"; the loss below ignores targets
        return (0.5 * (self.w[0] ** 2 - self.w[1] ** 2)).expand(x.shape[0], 1)


class _MeanLoss(nn.Module):
    def forward(self, out, y):
        return out.mean()


def _stub_factory(lam, direction):
    class _Stub:
        def __call__(self, *a, **kw):
            return self

        def eigenvalues(self, top_n=2):
            return [1.0, lam], [[torch.tensor([1.0, 0.0])], [torch.tensor(direction)]]

    return _Stub()


def test_fix3_nhe_is_a_negative_curvature_step_with_revert():
    x = torch.zeros(4, 1)
    y = torch.zeros(4, dtype=torch.long)
    # correct negative direction e1: loss must decrease and the step is kept
    model = _QuadraticSaddle()
    opt = SPRKD([model.w], base_optimizer=torch.optim.SGD([model.w], lr=0.1), loss_fn=_MeanLoss(),
                teacher_saddle_points=[torch.zeros(2)], nhe_step_mode="fixed", nhe_step_size=0.2,
                hessian_factory=_stub_factory(-1.0, [0.0, 1.0]))
    _MeanLoss()(model(x), y).backward()
    w_before = model.w.detach().clone()
    applied = opt._negative_hessian_eigenstep(group=opt.param_groups[0], model=model, data_batch=(x, y))
    assert applied
    delta = model.w.detach() - w_before
    assert abs(float(delta[0])) < 1e-7 and abs(abs(float(delta[1])) - 0.2) < 1e-6
    # moved away from 0 along w1 in the direction that lowers f (sign(g.v) = sign(-w1) < 0 -> +v)
    assert float(delta[1]) > 0
    c = opt.counters()
    assert c["nhe_taken"] == 1 and c["nhe_applied"] == 1 and c["nhe_reverted"] == 0

    # wrong "negative" direction e0 (actually positive curvature): loss rises -> reverted
    model2 = _QuadraticSaddle()
    opt2 = SPRKD([model2.w], base_optimizer=torch.optim.SGD([model2.w], lr=0.1), loss_fn=_MeanLoss(),
                 teacher_saddle_points=[torch.zeros(2)], nhe_step_mode="fixed", nhe_step_size=1.0,
                 hessian_factory=_stub_factory(-1.0, [1.0, 0.0]))
    _MeanLoss()(model2(x), y).backward()
    w2 = model2.w.detach().clone()
    applied2 = opt2._negative_hessian_eigenstep(group=opt2.param_groups[0], model=model2, data_batch=(x, y))
    assert not applied2
    assert torch.allclose(model2.w.detach(), w2)
    assert opt2.counters()["nhe_reverted"] == 1

    # no negative eigenvalue -> nothing happens, counted
    opt3 = SPRKD([model.w], base_optimizer=torch.optim.SGD([model.w], lr=0.1), loss_fn=_MeanLoss(),
                 teacher_saddle_points=[torch.zeros(2)], hessian_factory=_stub_factory(0.5, [0.0, 1.0]))
    assert not opt3._negative_hessian_eigenstep(group=opt3.param_groups[0], model=model, data_batch=(x, y))
    assert opt3.counters()["nhe_no_negative"] == 1


# (4) aggregation ----------------------------------------------------------------

def test_fix4_repository_top_k_and_aggregate_uses_best():
    repo = SaddlePointRepository(teacher_index=0, top_k=2)
    p = [torch.nn.Parameter(torch.zeros(2))]
    for loss, val in [(0.9, 9.0), (0.1, 1.0), (0.5, 5.0)]:
        p[0].data.fill_(val)
        repo.append(p, loss=loss)
    assert len(repo) == 2 and repo.n_dropped == 1
    assert sorted(repo.losses) == [0.1, 0.5]
    assert float(repo.best[0][0]) == 1.0

    other = SaddlePointRepository(teacher_index=1)
    p[0].data.fill_(3.0); other.append(p, loss=0.2)
    p[0].data.fill_(7.0); other.append(p, loss=0.8)   # last, but not best
    asr = aggregate_asr([repo, other])
    assert torch.allclose(asr[0], torch.full((2,), (1.0 + 3.0) / 2))
    asr_last = aggregate_asr([repo, other], select="last")
    assert torch.allclose(asr_last[0], torch.full((2,), (5.0 + 7.0) / 2))


# (5) gradient-norm gate and counters ---------------------------------------------

def test_fix5_grad_norm_gate_and_recording_counters(teacher_model, tiny_batch, cpu_loss):
    crit = SaddleCriterion(rule="magnitude", magnitude_threshold=7.0, max_grad_norm=1.0)
    assert is_strong_saddle_point([10.0, -8.0], crit, grad_norm=0.5)
    assert not is_strong_saddle_point([10.0, -8.0], crit, grad_norm=2.0)
    out = which_rules_fire([10.0, -8.0], crit, grad_norm=2.0)
    assert out["magnitude"] and not out["grad_gate"] and not out["fired"]

    class _Stub:
        def __call__(self, *a, **kw):
            return self

        def eigenvalues(self, top_n=4):
            return [10.0, -8.0], [None, None]

    x, y = tiny_batch
    base = torch.optim.SGD(teacher_model.parameters(), lr=1e-3)
    opt = SPRKD(teacher_model.parameters(), base_optimizer=base, loss_fn=cpu_loss, is_teacher=True,
                saddle_steps=1, saddle_criterion=SaddleCriterion(max_grad_norm=1e9), hessian_factory=_Stub())
    for _ in range(2):
        opt.zero_grad()
        loss = cpu_loss(teacher_model(x), y)
        loss.backward()
        opt.step(model=teacher_model, current_loss=loss.detach(), data_batch=(x, y))
    repo = opt.saddle_repository
    assert repo.n_checked == 2 and len(repo) == 2
    assert all(g == g and g > 0 for g in repo.grad_norms)   # finite, positive
    assert repo.steps == [1, 2]
    assert all(r["fired"] and r["rule"] == "magnitude" for r in repo.rules)
    assert opt.counters()["saddles_recorded"] == 2


# (6) depth-mismatched injection ---------------------------------------------------

def test_fix6_inject_state_list_handles_depth_mismatch():
    teacher = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.Conv2d(4, 8, 3))
    student = nn.Sequential(nn.Conv2d(3, 2, 3))
    with torch.no_grad():
        for p in teacher.parameters():
            p.fill_(1.0)
    state = [p.detach().clone() for p in teacher.parameters()]   # 4 tensors vs 2 in the student
    pairs = inject_state_list(student, state, teacher=teacher)
    assert pairs == [("0.weight", "0.weight"), ("0.bias", "0.bias")]
    assert torch.all(student[0].weight == 1.0) and torch.all(student[0].bias == 1.0)
    with pytest.raises(ValueError):
        inject_state_list(student, state)   # positional mode still requires equal counts


# (7) TinyImageNet validation normalisation ----------------------------------------

def test_fix7_tinyimagenet_val_is_normalised(tmp_path):
    PIL = pytest.importorskip("PIL")
    from PIL import Image

    from sprkd.data import TinyImageNetConfig, make_tinyimagenet_dataloaders

    root = tmp_path / "tiny"
    for cls in ("n01", "n02"):
        d = root / "train" / cls / "images"; d.mkdir(parents=True)
        Image.new("RGB", (64, 64), (255, 255, 255)).save(d / f"{cls}_0.JPEG")
    val = root / "val" / "images"; val.mkdir(parents=True)
    Image.new("RGB", (64, 64), (255, 255, 255)).save(val / "val_0.JPEG")
    Image.new("RGB", (64, 64), (255, 255, 255)).save(val / "val_1.JPEG")
    (root / "val" / "val_annotations.txt").write_text("val_0.JPEG\tn01\t0\t0\t0\t0\nval_1.JPEG\tn02\t0\t0\t0\t0\n")
    _, valid_loader, _ = make_tinyimagenet_dataloaders(TinyImageNetConfig(root=root, num_workers=0, batch_size=2))
    x, _ = next(iter(valid_loader))
    assert x.max() > 1.0, "a white image must map above 1 after mean/std normalisation"


# (8) transfer_via_graph ------------------------------------------------------------

def test_fix8_transfer_via_graph_raises_clear_error():
    with pytest.raises(NotImplementedError, match="not vendored"):
        transfer_via_graph(MalariaStudentCNN(), MalariaTeacherCNN())


def test_fix3_events_are_recorded():
    x = torch.zeros(4, 1); y = torch.zeros(4, dtype=torch.long)
    model = _QuadraticSaddle()
    opt = SPRKD([model.w], base_optimizer=torch.optim.SGD([model.w], lr=0.1), loss_fn=_MeanLoss(),
                teacher_saddle_points=[torch.zeros(2)], nhe_step_mode="fixed", nhe_step_size=0.2,
                hessian_factory=_stub_factory(-1.0, [0.0, 1.0]))
    _MeanLoss()(model(x), y).backward()
    opt._negative_hessian_eigenstep(group=opt.param_groups[0], model=model, data_batch=(x, y))
    assert len(opt.events) == 1 and opt.events[0]["kind"] == "nhe"
    assert opt.events[0]["post_loss"] < opt.events[0]["pre_loss"] and not opt.events[0]["reverted"]
