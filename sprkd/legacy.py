"""Backward-compatibility helpers for legacy SPRKD checkpoints.

The original ISEF 2023 SPRKD notebook saved fastai ``Learner`` objects whose
``opt`` attribute referenced a *notebook-level* ``SPRKD`` class living in
``__main__``. Loading those checkpoints from a fresh Python process raises::

    AttributeError: Can't get attribute 'SPRKD' on <module '__main__' ...>

This module provides :func:`enable_legacy_unpickling` which installs a
minimal stub class under ``__main__.SPRKD`` (and a few related names) so the
underlying *model* state_dict can be recovered intact. The stub does **not**
attempt to reconstruct the original optimizer logic - use the modern
:class:`sprkd.optimizer.SPRKD` for new training runs.
"""

from __future__ import annotations

import pickle
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN


class _LegacySPRKDStub:
    """Minimal stub that satisfies pickle when loading legacy fastai Learners.

    The stub is intentionally non-functional - it only exists to make
    ``torch.load`` succeed. After loading you should replace ``learner.opt``
    with a fresh :class:`sprkd.optimizer.SPRKD` if you intend to resume
    training.

    Note: we deliberately do *not* subclass ``torch.optim.Optimizer`` because
    that class wraps ``step`` with a metaclass that requires fully
    initialised parameter groups. Pickle only matches by class *name*, so a
    plain ``object`` subclass is sufficient here.
    """

    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        self.state: dict = {}
        self.param_groups: list[dict] = []
        self._sprkd_legacy_attrs = {"args": args, "kwargs": kwargs}

    def __setstate__(self, state: dict) -> None:  # noqa: D401
        self.__dict__.update(state)

    def step(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError(
            "Legacy SPRKD stub is non-functional. Re-create the optimizer "
            "with sprkd.optimizer.SPRKD before resuming training."
        )

    def zero_grad(self, *args, **kwargs):  # noqa: D401
        return None

    def state_dict(self) -> dict:  # noqa: D401
        return {"_legacy_stub": True}

    def load_state_dict(self, sd: dict) -> None:  # noqa: D401
        pass


_LEGACY_NAMES = (
    "SPRKD",
    "SPRKDLegacy",
    "SaddlePointOptimizer",
)


_STORAGE_ALIASES = (
    "FloatStorage",
    "HalfStorage",
    "LongStorage",
    "IntStorage",
    "ShortStorage",
    "CharStorage",
    "ByteStorage",
    "DoubleStorage",
    "BoolStorage",
)


def enable_legacy_pickle_cpu() -> None:
    """Redirect ``torch.cuda.*Storage`` to CPU storages when CUDA is absent.

    Legacy ISEF metric pickles were created with ``pickle.dump`` on a CUDA
    machine. Nested tensor blobs are reloaded via ``torch.load`` without an
    explicit ``map_location``. This shim makes both paths CPU-safe.
    """

    if torch.cuda.is_available():
        return
    for name in _STORAGE_ALIASES:
        cpu_cls = getattr(torch, name, None)
        if cpu_cls is not None and hasattr(torch.cuda, name):
            setattr(torch.cuda, name, cpu_cls)


@contextmanager
def legacy_pickle_cpu():
    """Context manager: CPU-safe ``pickle.load`` for legacy metric artifacts."""

    enable_legacy_pickle_cpu()
    real_load = torch.load

    def _cpu_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return real_load(*args, **kwargs)

    torch.load = _cpu_load  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.load = real_load  # type: ignore[method-assign]


def load_legacy_metrics_pkl(path: str | Path) -> Any:
    """Load a legacy ``METRICS/LOSSES AND ACCURACIES/*.pkl`` file on any device.

    Returns the original dict (typically ``{'TRAINING': ..., 'VALIDATION': ...}``).
    """

    path = Path(path)
    with legacy_pickle_cpu(), open(path, "rb") as f:
        return pickle.load(f)


def epoch_validation_series(
    metrics: dict,
    *,
    step_size: int = 323,
    split: str = "VALIDATION",
) -> Tuple[List[float], List[float]]:
    """Down-sample per-step validation metrics to per-epoch checkpoints.

    Experiment 1 reports validation every 323 training steps (paper Section 4.1).
    """

    block = metrics.get(split, {})
    losses = block.get("LOSSES", [])
    accs = block.get("ACCURACIES", [])
    if not losses or not accs:
        return [], []
    idx = range(step_size - 1, min(len(losses), len(accs)), step_size)
    epoch_losses = [float(losses[i]) for i in idx]
    epoch_accs = [float(accs[i]) for i in idx]
    return epoch_losses, epoch_accs


def enable_legacy_unpickling() -> None:
    """Install legacy SPRKD class stubs into ``__main__`` for unpickling.

    Idempotent: calling this more than once has no effect.
    """

    main_mod: Any = sys.modules.get("__main__")
    if main_mod is None:
        main_mod = types.ModuleType("__main__")
        sys.modules["__main__"] = main_mod

    for name in _LEGACY_NAMES:
        if not hasattr(main_mod, name):
            setattr(main_mod, name, _LegacySPRKDStub)


@contextmanager
def legacy_unpickling():
    """Context manager that ensures legacy unpickling stubs exist."""

    enable_legacy_unpickling()
    yield


def load_legacy_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load a legacy SPRKD ``.pth`` artifact and return the underlying object.

    Always installs the legacy stubs before loading; falls back to
    ``weights_only=False`` because the legacy artifacts are arbitrary
    Python objects, not pure tensor dicts.
    """

    enable_legacy_unpickling()
    return torch.load(path, map_location=map_location, weights_only=False)


def extract_state_dict(obj: Any) -> dict:
    """Best-effort extraction of a model ``state_dict`` from a legacy object.

    Handles the three formats observed in the released artifacts:

    * raw ``nn.Module`` -> ``module.state_dict()``;
    * fastai ``Learner`` -> ``learner.model.state_dict()``;
    * ``dict`` with a ``"model_state_dict"`` key -> that value.
    """

    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if hasattr(obj, "model") and isinstance(obj.model, nn.Module):
        return obj.model.state_dict()
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        # Possibly already a state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError(
        f"Could not extract a state_dict from a {type(obj).__name__} object."
    )


def load_legacy_student(path: str | Path) -> MalariaStudentCNN:
    """Load a legacy student checkpoint into a fresh :class:`MalariaStudentCNN`."""

    obj = load_legacy_checkpoint(path)
    sd = extract_state_dict(obj)
    model = MalariaStudentCNN()
    _strict_or_legacy_load(model, sd)
    return model


def load_legacy_teacher(path: str | Path) -> MalariaTeacherCNN:
    """Load a legacy teacher checkpoint into a fresh :class:`MalariaTeacherCNN`."""

    obj = load_legacy_checkpoint(path)
    sd = extract_state_dict(obj)
    model = MalariaTeacherCNN()
    _strict_or_legacy_load(model, sd)
    return model


def _strict_or_legacy_load(model: nn.Module, sd: dict) -> None:
    """Try strict loading; fall back to mapping ``Sequential`` keys to OO keys.

    The legacy notebook's ``nn.Sequential`` saves keys like ``"0.weight"``,
    while the new OO classes use names like ``"features.0.0.weight"``.
    """

    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        pass

    own_params: Iterable[nn.Parameter] = list(model.parameters())
    own_keys: list[str] = list(model.state_dict().keys())

    sd_tensors: list[torch.Tensor] = list(sd.values())
    if len(sd_tensors) != len(own_keys):
        raise RuntimeError(
            "Legacy state_dict has a different number of tensors than the "
            f"target model ({len(sd_tensors)} vs {len(own_keys)})."
        )

    new_sd = dict(zip(own_keys, sd_tensors))
    model.load_state_dict(new_sd, strict=True)
