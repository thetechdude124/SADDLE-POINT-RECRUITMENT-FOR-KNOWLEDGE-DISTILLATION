"""Tests that exercise the LFS-tracked artifacts shipped with the repo.

These tests are skipped automatically when the LFS objects have not been
pulled. To run them, install ``git-lfs`` and execute ``git lfs pull`` in the
repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sprkd.models import (
    MalariaStudentCNN,
    build_legacy_sequential_student,
    build_legacy_sequential_teacher,
)


pytestmark = pytest.mark.checkpoints


REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_lfs_pointer(path: Path) -> bool:
    """Return ``True`` iff ``path`` is still a tiny git-lfs pointer file."""

    try:
        with open(path, "rb") as f:
            head = f.read(64)
        return head.startswith(b"version https://git-lfs.github.com/spec/")
    except OSError:
        return True


def _require_real_file(path: Path):
    if not path.is_file():
        pytest.skip(f"{path} not present in working tree")
    if _is_lfs_pointer(path):
        pytest.skip(f"{path} is an LFS pointer; run `git lfs pull`")


def test_saddle_points_artifact_is_loadable():
    p = REPO_ROOT / "TRUE_MALARIA_ENSEMBLE_TEACHER_SADDLE_POINTS.pth"
    _require_real_file(p)
    saddles = torch.load(p, map_location="cpu", weights_only=False)
    assert isinstance(saddles, dict)
    # at least one teacher entry
    assert len(saddles) >= 1
    first = next(iter(saddles.values()))
    assert isinstance(first, list) and len(first) > 0
    snap = first[-1]
    assert isinstance(snap, list)
    for t in snap:
        assert isinstance(t, torch.Tensor)


def test_saddle_points_align_with_teacher_param_shapes():
    p = REPO_ROOT / "TRUE_MALARIA_ENSEMBLE_TEACHER_SADDLE_POINTS.pth"
    _require_real_file(p)
    saddles = torch.load(p, map_location="cpu", weights_only=False)
    snap = next(iter(saddles.values()))[-1]
    teacher = build_legacy_sequential_teacher()
    expected = [p.shape for p in teacher.parameters()]
    actual = [t.shape for t in snap]
    assert expected == actual, f"shape mismatch:\n expected={expected}\n actual={actual}"


def test_test_set_artifact_is_loadable():
    p = REPO_ROOT / "TESTSET.pth"
    _require_real_file(p)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    # Could be tensor, dict, or torch.utils.data object - just sanity-check shape
    assert obj is not None


def test_sprkd_student_checkpoint_has_correct_param_count():
    """Loads the released SPRKD student model and confirms 6,430 parameters.

    The legacy artifact is a pickled fastai ``Learner`` whose ``opt`` slot
    references the notebook-level ``SPRKD`` class; we install the
    compatibility stub via :func:`sprkd.legacy.enable_legacy_unpickling`
    before loading.
    """

    p = REPO_ROOT / "MODELS" / "SPRKD_MALARIA.pth"
    _require_real_file(p)

    try:
        from sprkd.legacy import (  # noqa: F401 - imported for side effects
            enable_legacy_unpickling,
            extract_state_dict,
            load_legacy_checkpoint,
        )
    except Exception as e:  # pragma: no cover
        pytest.skip(f"legacy module unavailable: {e}")

    try:
        obj = load_legacy_checkpoint(p)
    except (ImportError, AttributeError) as e:
        # fastai may not be importable in the test env; that's okay - the
        # released artifacts include fastai Learners. We only assert what we
        # *can* load.
        pytest.skip(f"could not unpickle legacy artifact: {e}")

    sd = extract_state_dict(obj)
    student = MalariaStudentCNN()
    from sprkd.legacy import _strict_or_legacy_load

    _strict_or_legacy_load(student, sd)
    assert sum(t.numel() for t in student.parameters()) == 6_430


def test_pyhessian_eig_artifact_loads():
    p = REPO_ROOT / "SPRKD_10_PYHESSIAN_EIGS.pth"
    _require_real_file(p)
    obj = torch.load(p, map_location="cpu", weights_only=False)
    assert obj is not None


def test_load_legacy_student_via_public_api_runs_inference():
    """End-to-end: legacy artifact -> public loader -> forward pass."""

    p = REPO_ROOT / "MODELS" / "SPRKD_MALARIA.pth"
    _require_real_file(p)
    try:
        from sprkd import load_legacy_student
    except Exception as e:  # pragma: no cover
        pytest.skip(f"public legacy API unavailable: {e}")

    try:
        model = load_legacy_student(p)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"legacy artifact not loadable in this env: {e}")

    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    # outputs are softmax probabilities (model definition includes softmax)
    assert torch.allclose(out.sum(dim=1), torch.ones(2), atol=1e-4)
