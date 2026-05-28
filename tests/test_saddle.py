"""Tests for saddle-point detection and ASR aggregation."""

import math

import pytest
import torch

from sprkd.saddle import (
    SaddleCriterion,
    SaddlePointRepository,
    aggregate_asr,
    is_strong_saddle_point,
)


# ---------------------------------------------------------------------------
# is_strong_saddle_point
# ---------------------------------------------------------------------------

def test_pure_minimum_is_not_a_saddle():
    """Default rule is 'magnitude' (canonical Colab): no negatives -> False."""

    eigs = [10.0, 8.0, 5.0, 1.0]
    assert is_strong_saddle_point(eigs) is False


def test_default_rule_is_magnitude_canonical_colab():
    """Canonical Colab: ``abs(sum(neg_eigs)) >= 7`` is the only test."""

    assert is_strong_saddle_point([10.0, -8.0]) is True   # |sum_neg|=8 >= 7
    assert is_strong_saddle_point([10.0, -6.5]) is False  # |sum_neg|=6.5 < 7


def test_pure_maximum_passes_magnitude_rule():
    eigs = [-10.0, -8.0, -5.0, -1.0]
    assert is_strong_saddle_point(eigs) is True  # |sum_neg| = 24 >= 7


def test_classical_saddle_qualifies_under_both_rules():
    eigs = [10.0, -9.0, 1.0, -1.0]
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="magnitude")) is True
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="ratio")) is True
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="both")) is True


def test_ratio_rule_excludes_weak_negatives():
    eigs = [100.0, 50.0, -1.0]
    crit = SaddleCriterion(rule="ratio", alpha=0.4)
    assert is_strong_saddle_point(eigs, crit) is False


def test_magnitude_rule_with_threshold_zero_passes_anything_with_neg():
    eigs = [10.0, -0.5]
    crit = SaddleCriterion(rule="magnitude", magnitude_threshold=0.0)
    assert is_strong_saddle_point(eigs, crit) is True


def test_both_rule_requires_both_conditions():
    """Eigs satisfying ratio but failing magnitude must be rejected by 'both'."""

    eigs = [10.0, -5.0]  # ratio: 5 >= 0.4*10 OK, magnitude: 5 < 7
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="both")) is False
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="ratio")) is True


def test_unknown_rule_raises():
    eigs = [10.0, -5.0]
    crit = SaddleCriterion(rule="not-a-rule")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown saddle rule"):
        is_strong_saddle_point(eigs, crit)


def test_require_negative_flag():
    eigs = [10.0, 5.0]
    crit = SaddleCriterion(
        rule="ratio",
        alpha=0.0,
        magnitude_threshold=0.0,
        require_negative_eigenvalue=False,
    )
    assert is_strong_saddle_point(eigs, crit) is True
    crit2 = SaddleCriterion(
        rule="ratio",
        alpha=0.0,
        magnitude_threshold=0.0,
        require_negative_eigenvalue=True,
    )
    assert is_strong_saddle_point(eigs, crit2) is False


# ---------------------------------------------------------------------------
# SaddlePointRepository
# ---------------------------------------------------------------------------

def test_repository_stores_cpu_clones(student_model):
    repo = SaddlePointRepository(teacher_index=0)
    repo.append(student_model.parameters(), loss=0.5)
    assert len(repo) == 1
    snap = repo.snapshots[0]
    assert all(t.device.type == "cpu" for t in snap)
    # mutating the model parameters should not affect stored snapshot
    for p in student_model.parameters():
        with torch.no_grad():
            p.add_(1.0)
    snap2 = [p.detach().cpu().clone() for p in student_model.parameters()]
    for a, b in zip(repo.snapshots[0], snap2):
        assert not torch.allclose(a, b)


def test_repository_best_returns_lowest_loss(student_model):
    repo = SaddlePointRepository(teacher_index=0)
    repo.append(student_model.parameters(), loss=0.9)
    repo.append(student_model.parameters(), loss=0.1)
    repo.append(student_model.parameters(), loss=0.5)
    best = repo.best
    assert all(torch.allclose(a, b) for a, b in zip(best, repo.snapshots[1]))


def test_repository_best_falls_back_to_last_when_no_losses(student_model):
    repo = SaddlePointRepository(teacher_index=0)
    repo.append(student_model.parameters(), loss=float("nan"))
    repo.append(student_model.parameters(), loss=float("nan"))
    best = repo.best
    assert all(torch.allclose(a, b) for a, b in zip(best, repo.snapshots[-1]))


# ---------------------------------------------------------------------------
# aggregate_asr
# ---------------------------------------------------------------------------

def test_aggregate_asr_averages_across_teachers():
    a = [torch.ones(3, 3), torch.ones(2)]
    b = [torch.zeros(3, 3), torch.zeros(2)]
    asr = aggregate_asr([[a], [b]])
    assert torch.allclose(asr[0], torch.full((3, 3), 0.5))
    assert torch.allclose(asr[1], torch.full((2,), 0.5))


def test_aggregate_asr_uses_last_snapshot():
    a_old = [torch.zeros(2, 2)]
    a_new = [torch.ones(2, 2) * 4]
    b_only = [torch.ones(2, 2) * 2]
    asr = aggregate_asr([[a_old, a_new], [b_only]])
    assert torch.allclose(asr[0], torch.full((2, 2), 3.0))


def test_aggregate_asr_raises_on_empty():
    with pytest.raises(ValueError):
        aggregate_asr([])


def test_aggregate_asr_respects_device_arg():
    a = [torch.ones(2, 2)]
    asr = aggregate_asr([[a]], device=torch.device("cpu"))
    assert asr[0].device.type == "cpu"
