"""Tests for the legacy compatibility shim."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sprkd.legacy import (
    _LegacySPRKDStub,
    enable_legacy_unpickling,
    epoch_validation_series,
    extract_state_dict,
    legacy_unpickling,
    load_legacy_metrics_pkl,
)
from sprkd.models import MalariaStudentCNN, build_legacy_sequential_student


def test_enable_legacy_unpickling_installs_stub():
    enable_legacy_unpickling()
    main_mod = sys.modules["__main__"]
    assert hasattr(main_mod, "SPRKD")
    assert main_mod.SPRKD is _LegacySPRKDStub


def test_enable_legacy_unpickling_idempotent():
    enable_legacy_unpickling()
    enable_legacy_unpickling()  # second call must not error


def test_legacy_stub_is_picklable_class():
    """The stub need not be a torch.optim.Optimizer; it just needs the same name."""

    stub = _LegacySPRKDStub()
    assert stub.__class__.__name__ == "_LegacySPRKDStub"
    assert isinstance(stub, _LegacySPRKDStub)


def test_legacy_stub_step_raises():
    stub = _LegacySPRKDStub()
    with pytest.raises(RuntimeError, match="non-functional"):
        stub.step()


def test_legacy_unpickling_context_manager():
    with legacy_unpickling():
        assert "SPRKD" in vars(sys.modules["__main__"])


def test_extract_state_dict_from_module():
    model = MalariaStudentCNN()
    sd = extract_state_dict(model)
    assert set(sd.keys()) == set(model.state_dict().keys())


def test_extract_state_dict_from_state_dict():
    model = MalariaStudentCNN()
    sd_in = model.state_dict()
    sd_out = extract_state_dict(sd_in)
    assert sd_out is sd_in


def test_extract_state_dict_from_dict_with_key():
    model = MalariaStudentCNN()
    payload = {"model_state_dict": model.state_dict(), "extra": 1}
    sd = extract_state_dict(payload)
    assert sd is payload["model_state_dict"]


def test_extract_state_dict_invalid_type_raises():
    with pytest.raises(ValueError):
        extract_state_dict(42)


def test_load_legacy_metrics_pkl_returns_validation_block():
    path = (
        Path(__file__).resolve().parent.parent
        / "METRICS"
        / "LOSSES AND ACCURACIES"
        / "500_SPRKD_LOSSES.pkl"
    )
    if not path.is_file():
        pytest.skip("metrics pickle not present")
    metrics = load_legacy_metrics_pkl(path)
    assert "VALIDATION" in metrics
    losses, accs = epoch_validation_series(metrics, step_size=323)
    assert len(losses) == len(accs) > 0


def test_strict_or_legacy_load_remaps_keys():
    """Sequential checkpoints (legacy) -> OO model (modern)."""

    legacy = build_legacy_sequential_student()
    legacy_sd = legacy.state_dict()
    modern = MalariaStudentCNN()

    from sprkd.legacy import _strict_or_legacy_load

    _strict_or_legacy_load(modern, legacy_sd)
    # After loading, parameters should match positionally
    for a, b in zip(legacy.parameters(), modern.parameters()):
        assert torch.allclose(a, b)
