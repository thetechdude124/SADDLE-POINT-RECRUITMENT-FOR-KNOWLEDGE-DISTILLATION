"""Tests for the SPRKD optimizer modes."""

import pytest
import torch
import torch.nn as nn

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.optimizer import SPRKD
from sprkd.saddle import SaddleCriterion


class _StubHessian:
    """Test double for ``pyhessian.hessian`` so we can run optimizer tests on CPU."""

    def __init__(self, eigenvalues_seq=None):
        self.eigenvalues_seq = list(eigenvalues_seq or [[10.0, -9.0, 1.0, -1.0]])
        self._call_count = 0

    def __call__(self, model, criterion, data, use_cuda):
        # ignore inputs - this is an opaque stub
        return self

    def eigenvalues(self, top_n=4):
        idx = min(self._call_count, len(self.eigenvalues_seq) - 1)
        self._call_count += 1
        eigs = self.eigenvalues_seq[idx]
        return eigs[:top_n], [torch.zeros(1) for _ in range(top_n)]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_teacher_and_control_modes_are_mutually_exclusive(teacher_model, cpu_loss):
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        SPRKD(
            teacher_model.parameters(),
            base_optimizer=base,
            loss_fn=cpu_loss,
            is_teacher=True,
            is_control=True,
        )


def test_student_mode_requires_saddle_points(student_model, cpu_loss):
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        SPRKD(student_model.parameters(), base_optimizer=base, loss_fn=cpu_loss)


def test_invalid_epsilon_rejected(student_model, cpu_loss):
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    with pytest.raises(ValueError):
        SPRKD(
            student_model.parameters(),
            base_optimizer=base,
            loss_fn=cpu_loss,
            teacher_saddle_points=asr,
            epsilon=-0.1,
        )


# ---------------------------------------------------------------------------
# Control mode
# ---------------------------------------------------------------------------

def test_control_mode_passes_through_to_base_optimizer(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.SGD(student_model.parameters(), lr=1e-2)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_control=True,
    )
    p0 = [p.detach().clone() for p in student_model.parameters()]
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    sprkd.step(model=student_model, current_loss=loss.detach(), data_batch=(x, y))
    p1 = list(student_model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(p0, p1))


def test_step_without_model_uses_base_optimizer(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.SGD(student_model.parameters(), lr=1e-2)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
    )
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    p0 = [p.detach().clone() for p in student_model.parameters()]
    sprkd.step()  # no model context
    p1 = list(student_model.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(p0, p1))


# ---------------------------------------------------------------------------
# Teacher mode
# ---------------------------------------------------------------------------

def test_teacher_mode_records_saddle(teacher_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    stub = _StubHessian([[10.0, -9.0, 1.0, -1.0]])
    sprkd = SPRKD(
        teacher_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_teacher=True,
        saddle_steps=1,
        saddle_criterion=SaddleCriterion(rule="magnitude", magnitude_threshold=7.0),
        hessian_factory=stub,
    )
    logits = teacher_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    sprkd.step(model=teacher_model, current_loss=loss.detach(), data_batch=(x, y))
    assert len(sprkd.saddle_repository) == 1


def test_teacher_mode_skips_non_saddle(teacher_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    stub = _StubHessian([[10.0, 8.0, 5.0, 1.0]])  # pure positive => not a saddle
    sprkd = SPRKD(
        teacher_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_teacher=True,
        saddle_steps=1,
        hessian_factory=stub,
    )
    logits = teacher_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    sprkd.step(model=teacher_model, current_loss=loss.detach(), data_batch=(x, y))
    assert len(sprkd.saddle_repository) == 0


def test_teacher_mode_respects_saddle_steps(teacher_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    stub = _StubHessian([[10.0, -9.0, 1.0, -1.0]] * 5)
    sprkd = SPRKD(
        teacher_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_teacher=True,
        saddle_steps=3,
        hessian_factory=stub,
    )
    for _ in range(3):
        logits = teacher_model(x)
        loss = cpu_loss(logits, y)
        loss.backward()
        sprkd.step(model=teacher_model, current_loss=loss.detach(), data_batch=(x, y))
    # only step #3 triggers a check
    assert len(sprkd.saddle_repository) == 1


# ---------------------------------------------------------------------------
# Student mode
# ---------------------------------------------------------------------------

def test_student_targeting_initially_active(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    asr = [torch.randn_like(p) for p in student_model.parameters()]
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
        saddle_steps=None,
    )
    assert sprkd.at_asr() is False
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    sprkd.step(model=student_model, current_loss=loss.detach(), data_batch=(x, y))
    # at least one parameter should be flagged for ASR-seeking
    assert any(sprkd._allow_targeting.values())


def test_student_at_asr_when_targets_are_satisfied(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    # ASR equals current parameters -> targeting deactivates immediately
    asr = [p.detach().clone() for p in student_model.parameters()]
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
        saddle_steps=None,
        epsilon=10.0,  # very loose - everything is already 'at ASR'
    )
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    sprkd.step(model=student_model, current_loss=loss.detach(), data_batch=(x, y))
    assert sprkd.at_asr() is True


def test_state_dict_carries_sprkd_extras(student_model, cpu_loss):
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
    )
    sd = sprkd.state_dict()
    assert "sprkd_extra" in sd
    extras = sd["sprkd_extra"]
    for key in ("step_count", "allow_targeting", "cooldown", "n_nhe_taken"):
        assert key in extras


# ---------------------------------------------------------------------------
# Canonical-Colab parity (new in v0.1.0)
# ---------------------------------------------------------------------------

def test_saddle_step_limit_disables_tracking_after_threshold(
    teacher_model, tiny_batch, cpu_loss
):
    x, y = tiny_batch
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    stub = _StubHessian([[10.0, -9.0, 1.0, -1.0]] * 10)
    sprkd = SPRKD(
        teacher_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_teacher=True,
        saddle_steps=1,
        saddle_step_limit=2,
        hessian_factory=stub,
    )
    for _ in range(5):
        logits = teacher_model(x)
        loss = cpu_loss(logits, y)
        loss.backward()
        sprkd.step(model=teacher_model, current_loss=loss.detach(), data_batch=(x, y))
    # only the step at t=1 is < limit=2; subsequent ones are skipped
    assert len(sprkd.saddle_repository) == 1


def test_invalid_saddle_step_limit_rejected(teacher_model, cpu_loss):
    base = torch.optim.Adam(teacher_model.parameters(), lr=1e-3)
    with pytest.raises(ValueError):
        SPRKD(
            teacher_model.parameters(),
            base_optimizer=base,
            loss_fn=cpu_loss,
            is_teacher=True,
            saddle_step_limit=0,
        )


def test_invalid_nhe_step_mode_rejected(student_model, cpu_loss):
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    with pytest.raises(ValueError):
        SPRKD(
            student_model.parameters(),
            base_optimizer=base,
            loss_fn=cpu_loss,
            teacher_saddle_points=asr,
            nhe_step_mode="quadratic",
        )


def test_invalid_pgd_perturb_variance_rejected(student_model, cpu_loss):
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    with pytest.raises(ValueError):
        SPRKD(
            student_model.parameters(),
            base_optimizer=base,
            loss_fn=cpu_loss,
            teacher_saddle_points=asr,
            pgd_perturb_variance=-0.5,
        )


def test_default_pgd_variance_matches_paper_section_3_3_2(student_model, cpu_loss):
    """Paper Section 3.3.2 specifies xi ~ N(0, 0.1)."""

    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
    )
    assert sprkd.param_groups[0]["pgd_perturb_variance"] == 0.1


def test_default_nhe_mode_is_adaptive(student_model, cpu_loss):
    """Canonical Colab: ``weight = 1 / largest_negative_eigenvalue``."""

    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
    )
    assert sprkd.param_groups[0]["nhe_step_mode"] == "adaptive"


def test_default_saddle_threshold_matches_canonical_colab(student_model, cpu_loss):
    """Canonical Colab: ``abs(sum(neg_eigs)) >= 7`` (`magnitude` rule)."""

    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
    )
    assert sprkd.saddle_criterion.rule == "magnitude"
    assert sprkd.saddle_criterion.magnitude_threshold == 7.0


def test_nhe_adaptive_uses_inverse_eigenvalue_weight(student_model, tiny_batch, cpu_loss):
    """Verify NHE step magnitude scales as ``1 / |lambda_neg|``."""

    x, y = tiny_batch

    class _Stub:
        def __init__(self, lam):
            self.lam = lam

        def __call__(self, *a, **kw):
            return self

        def eigenvalues(self, top_n=2):
            # one large positive, one large negative; eigenvectors of ones
            vec_layers = [torch.ones_like(p) for p in student_model.parameters()]
            return [self.lam, -self.lam], [vec_layers, vec_layers]

    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    base = torch.optim.SGD(student_model.parameters(), lr=1.0)
    lam = 4.0
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
        nhe_step_mode="adaptive",
        max_nhe_steps=10,
        hessian_factory=_Stub(lam),
    )
    # Manually invoke NHE to isolate behaviour from the PGD trigger logic
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()

    p_before = [p.detach().clone() for p in student_model.parameters()]
    grads_before = [p.grad.detach().clone() for p in student_model.parameters()]
    sprkd._negative_hessian_eigenstep(
        group=sprkd.param_groups[0],
        model=student_model,
        data_batch=(x, y),
    )
    expected_weight = 1.0 / lam  # |lambda_neg| = lam
    for p_now, p_old, g in zip(student_model.parameters(), p_before, grads_before):
        # eigenvector layers are all-ones, so step = grad * 1 * 1 = grad
        delta = (p_now - p_old).detach()
        assert torch.allclose(delta, -expected_weight * g, atol=1e-6)


def test_nhe_fixed_uses_constant_weight(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch

    class _Stub:
        def __call__(self, *a, **kw):
            return self

        def eigenvalues(self, top_n=2):
            vec_layers = [torch.ones_like(p) for p in student_model.parameters()]
            return [5.0, -5.0], [vec_layers, vec_layers]

    asr = [torch.zeros_like(p) for p in student_model.parameters()]
    base = torch.optim.SGD(student_model.parameters(), lr=1.0)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=asr,
        nhe_step_mode="fixed",
        nhe_step_size=0.1,
        max_nhe_steps=10,
        hessian_factory=_Stub(),
    )
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    p_before = [p.detach().clone() for p in student_model.parameters()]
    grads_before = [p.grad.detach().clone() for p in student_model.parameters()]
    sprkd._negative_hessian_eigenstep(
        group=sprkd.param_groups[0],
        model=student_model,
        data_batch=(x, y),
    )
    for p_now, p_old, g in zip(student_model.parameters(), p_before, grads_before):
        delta = (p_now - p_old).detach()
        assert torch.allclose(delta, -0.1 * g, atol=1e-6)
