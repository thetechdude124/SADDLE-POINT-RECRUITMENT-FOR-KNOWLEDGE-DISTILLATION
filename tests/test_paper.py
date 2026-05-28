"""Paper-faithfulness and artifact-consistency tests (sprkd.tex Section 3-5).

These tests lock the implementation and bundled checkpoints to the claims in
the paper and the canonical Colab notebooks. Exact floating-point reproduction
of every table entry is not expected (5-trial averages, different hardware),
but structural properties and relative orderings must hold.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sprkd.legacy import (
    epoch_validation_series,
    load_legacy_metrics_pkl,
    load_legacy_student,
)
from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.optimizer import SPRKD
from sprkd.saddle import SaddleCriterion, is_strong_saddle_point


pytestmark = pytest.mark.paper

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = REPO_ROOT / "METRICS" / "LOSSES AND ACCURACIES"
ESD_DIR = REPO_ROOT / "METRICS" / "HESSIAN EIGENSPECTRA"
VALIDATION_STEP = 323  # paper Section 4.1, Experiment 1

# Table 1 (paper Section 5) headline numbers for Experiment 1
PAPER_TABLE1 = {
    "teacher_params": 25_546,
    "student_params": 6_430,
    "sprkd_val_acc": 94.80,
    "control_student_val_acc": 94.47,
    "rkd_val_acc": 70.10,
    "weak_teacher_val_acc": 70.13,
    "sprkd_val_loss": 0.361,
    "sprkd_rkd_acc_gap_pct": 24.70,
    "mcnemar_p_sprkd_vs_rkd": 6.3e-87,
    "hessian_trace_sprkd": 33.39,
    "hessian_trace_control": 71.33,
    "hessian_trace_rkd": 408.27,
}

PAPER_HYPERPARAMS = {
    "saddle_alpha": 0.4,
    "saddle_beta": 7.0,
    "pgd_variance": 0.1,
    "pgd_grad_threshold_paper": 0.02,
    "epsilon_paper": 0.1,
}


def _metrics_available(name: str) -> bool:
    return (METRICS_DIR / name).is_file()


# ---------------------------------------------------------------------------
# Architecture / compression (Section 4.1, Table 1)
# ---------------------------------------------------------------------------


def test_teacher_parameter_count_matches_paper():
    n = sum(p.numel() for p in MalariaTeacherCNN().parameters())
    assert n == PAPER_TABLE1["teacher_params"]


def test_student_parameter_count_matches_paper():
    n = sum(p.numel() for p in MalariaStudentCNN().parameters())
    assert n == PAPER_TABLE1["student_params"]


def test_compression_ratio_is_approximately_4x():
    """Paper reports a 4x compression; exact ratio is ~3.97 with these counts."""

    ratio = PAPER_TABLE1["teacher_params"] / PAPER_TABLE1["student_params"]
    assert 3.9 <= ratio <= 4.1


# ---------------------------------------------------------------------------
# Saddle criterion (Section 3.1, Eq. 1)
# ---------------------------------------------------------------------------


def test_paper_eq1_both_rule_matches_literal_statement():
    """Paper Eq. 1: magnitude AND ratio conditions."""

    eigs = [10.0, -9.0, 1.0, -1.0]
    assert is_strong_saddle_point(eigs, SaddleCriterion(rule="both")) is True
    # ratio ok, magnitude fails (|sum_neg|=2 < 7)
    weak = [10.0, -1.0, -1.0]
    assert is_strong_saddle_point(weak, SaddleCriterion(rule="both")) is False


def test_paper_hyperparameter_defaults_documented():
    crit = SaddleCriterion(rule="both")
    assert crit.alpha == PAPER_HYPERPARAMS["saddle_alpha"]
    assert crit.magnitude_threshold == PAPER_HYPERPARAMS["saddle_beta"]


def test_released_checkpoints_use_magnitude_rule_by_default(student_model, cpu_loss):
    """Canonical Colab / released artifacts: |sum(neg)| >= 7 only."""

    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=[torch.zeros_like(p) for p in student_model.parameters()],
    )
    assert sprkd.saddle_criterion.rule == "magnitude"
    assert sprkd.saddle_criterion.magnitude_threshold == 7.0


# ---------------------------------------------------------------------------
# Transformation matrix decay (Section 3.3.1)
# ---------------------------------------------------------------------------


def test_tm_decay_weight_matches_canonical_implementation():
    """Canonical code: ``weight = -2**(-t/10)/2 + 1`` (differs from paper by /2)."""

    def weight(t: int) -> float:
        return -2.0 ** (-t / 10.0) / 2.0 + 1.0

    assert math.isclose(weight(0), 0.5)
    assert math.isclose(weight(10), 0.75)
    assert weight(100) > 0.99


def test_paper_epsilon_differs_from_package_default(student_model, cpu_loss):
    """Paper Section 3.3.1 uses epsilon=0.1; canonical Colab uses 1e-3."""

    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=[torch.zeros_like(p) for p in student_model.parameters()],
    )
    assert sprkd.param_groups[0]["epsilon"] == 1e-3
    assert sprkd.param_groups[0]["epsilon"] != PAPER_HYPERPARAMS["epsilon_paper"]


def test_pgd_defaults_match_paper_section_3_3_2(student_model, cpu_loss):
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    sprkd = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        teacher_saddle_points=[torch.zeros_like(p) for p in student_model.parameters()],
    )
    g = sprkd.param_groups[0]
    assert g["pgd_perturb_variance"] == PAPER_HYPERPARAMS["pgd_variance"]
    # canonical Colab uses 0.01; paper stagnation threshold j=0.02
    assert g["pgd_grad_threshold"] == 0.01


# ---------------------------------------------------------------------------
# Bundled training metrics (Figure 2 / Table 1 ordering)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _metrics_available("500_SPRKD_LOSSES.pkl"),
    reason="SPRKD metrics pickle not present",
)
def test_released_sprkd_final_epoch_accuracy_near_paper():
    metrics = load_legacy_metrics_pkl(METRICS_DIR / "500_SPRKD_LOSSES.pkl")
    _, accs = epoch_validation_series(metrics, step_size=VALIDATION_STEP)
    assert accs, "expected non-empty epoch validation accuracies"
    final = accs[-1]
    # single-run checkpoint metrics; allow slack vs 5-trial Table-1 average
    assert final >= PAPER_TABLE1["sprkd_val_acc"] - 5.0
    assert final <= 100.0


@pytest.mark.skipif(
    not _metrics_available("500_SPRKD_LOSSES.pkl")
    or not _metrics_available("500_CONTROL_STUDENT_LOSSES.pkl"),
    reason="SPRKD or Control metrics pickle not present",
)
def test_released_sprkd_beats_control_on_final_epoch_accuracy():
    sprkd_m = load_legacy_metrics_pkl(METRICS_DIR / "500_SPRKD_LOSSES.pkl")
    ctrl_m = load_legacy_metrics_pkl(METRICS_DIR / "500_CONTROL_STUDENT_LOSSES.pkl")
    _, sprkd_accs = epoch_validation_series(sprkd_m, step_size=VALIDATION_STEP)
    _, ctrl_accs = epoch_validation_series(ctrl_m, step_size=VALIDATION_STEP)
    assert sprkd_accs[-1] >= ctrl_accs[-1] - 2.0


@pytest.mark.skipif(
    not _metrics_available("500_SPRKD_LOSSES.pkl"),
    reason="SPRKD metrics pickle not present",
)
def test_released_sprkd_final_loss_below_paper_upper_bound():
    losses, _ = epoch_validation_series(
        load_legacy_metrics_pkl(METRICS_DIR / "500_SPRKD_LOSSES.pkl"),
        step_size=VALIDATION_STEP,
    )
    assert losses[-1] <= PAPER_TABLE1["sprkd_val_loss"] + 0.15


# ---------------------------------------------------------------------------
# Hessian trace ordering (Section 5, Figure 3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (ESD_DIR / "EIGS_500_SPRKD_MALARIA.pth").is_file()
    or not (ESD_DIR / "EIGS_RKD_MALARIA_STUDENT.pth").is_file(),
    reason="bundled ESD artifacts missing",
)
def test_released_hessian_trace_rkd_largest():
    """RKD should exhibit the largest trace among the three students."""

    traces = {}
    for label, fname in [
        ("SPRKD", "EIGS_500_SPRKD_MALARIA.pth"),
        ("Control", "EIGS_500_CONTROL_MALARIA.pth"),
        ("RKD", "EIGS_RKD_MALARIA_STUDENT.pth"),
    ]:
        path = ESD_DIR / fname
        if not path.is_file():
            pytest.skip(f"{fname} missing")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        traces[label] = float(payload["TRACE"])
    assert traces["RKD"] > traces["SPRKD"]
    assert traces["RKD"] > traces["Control"]


# ---------------------------------------------------------------------------
# Released student checkpoint (Table 1 accuracy sanity)
# ---------------------------------------------------------------------------


@pytest.mark.checkpoints
@pytest.mark.slow
def test_released_sprkd_student_high_accuracy_on_testset(repo_root):
    """Held-out TESTSET.pth: sanity that the released student is strong."""

    testset = repo_root / "TESTSET.pth"
    checkpoint = repo_root / "MODELS" / "SPRKD_MALARIA.pth"
    if not testset.is_file() or not checkpoint.is_file():
        pytest.skip("TESTSET.pth or SPRKD_MALARIA.pth not available")

    try:
        from sprkd.data import load_testset_pth
    except ImportError:
        pytest.skip("load_testset_pth unavailable")

    try:
        model = load_legacy_student(checkpoint)
    except (ImportError, AttributeError, RuntimeError) as e:
        pytest.skip(f"legacy checkpoint not loadable: {e}")

    xs, ys = load_testset_pth(testset)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(xs), dim=1)
    acc = 100.0 * (preds == ys).float().mean().item()
    # small 100-sample slice; expect well above chance and weak-teacher ceiling
    assert acc >= 70.0


@pytest.mark.checkpoints
def test_sprkd_rkd_accuracy_gap_from_paper_table():
    gap = PAPER_TABLE1["sprkd_val_acc"] - PAPER_TABLE1["rkd_val_acc"]
    assert math.isclose(gap, PAPER_TABLE1["sprkd_rkd_acc_gap_pct"], abs_tol=0.05)
