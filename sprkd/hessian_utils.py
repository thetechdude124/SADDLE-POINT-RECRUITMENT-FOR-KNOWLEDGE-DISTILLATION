"""Safe wrappers around PyHessian.

``pyhessian.hessian`` has two side effects that silently corrupt a training loop:

* its constructor calls ``model.eval()`` and never restores train mode, so dropout and
  BatchNorm switch off for the rest of the epoch;
* ``eigenvalues`` / ``trace`` / ``density`` call ``model.zero_grad()`` (which sets every
  ``.grad`` to ``None`` in torch >= 2.0), so any optimizer logic that reads gradients after
  a Hessian call sees nothing.

Every Hessian computation in this package goes through :func:`preserve_model_state`, which
snapshots the training flag and all gradients before the call and restores them after.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Tuple

import torch
import torch.nn as nn


@contextmanager
def preserve_model_state(model: nn.Module) -> Iterator[None]:
    """Restore ``model.training`` and every parameter's ``.grad`` on exit."""

    was_training = model.training
    saved_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    try:
        yield
    finally:
        model.train(was_training)
        for p, g in zip(model.parameters(), saved_grads):
            p.grad = None if g is None else g.to(p.device)


@contextmanager
def hessian_compatible(
    model: nn.Module, data_batch: Optional[tuple]
) -> Iterator[Tuple[nn.Module, Optional[tuple], bool]]:
    """Yield ``(model, batch, use_cuda)`` safe for PyHessian, with state preserved.

    PyHessian supports only ``cuda`` and ``cpu``. On MPS the model and batch are moved to
    CPU for the duration of the call and moved back afterwards. Train/eval mode and
    gradients are restored on exit regardless of device.
    """

    original = next(model.parameters()).device
    moved = False
    with preserve_model_state(model):
        try:
            if original.type == "mps":
                model.to("cpu")
                moved = True
                if data_batch is not None and isinstance(data_batch, (tuple, list)):
                    data_batch = tuple(d.to("cpu") if hasattr(d, "to") else d for d in data_batch)
                use_cuda = False
            else:
                use_cuda = original.type == "cuda"
            yield model, data_batch, use_cuda
        finally:
            if moved:
                model.to(original)


def default_hessian_factory(model: nn.Module, criterion: nn.Module, data: tuple, use_cuda: bool):
    """Build a ``pyhessian.hessian`` object (imported lazily)."""

    from pyhessian import hessian as PyHessian

    return PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)


def top_eigenpairs(
    model: nn.Module,
    criterion: nn.Module,
    data_batch: tuple,
    top_n: int,
    factory: Optional[Callable[..., Any]] = None,
):
    """Top-``top_n`` Hessian eigenvalues/eigenvectors with model state preserved."""

    factory = factory or default_hessian_factory
    with hessian_compatible(model, data_batch) as (m, batch, use_cuda):
        hess = factory(m, criterion, batch, use_cuda)
        eigenvalues, eigenvectors = hess.eigenvalues(top_n=top_n)
    return eigenvalues, eigenvectors


@torch.no_grad()
def batch_loss(model: nn.Module, criterion: nn.Module, data_batch: tuple) -> float:
    """Loss of ``model`` on ``data_batch`` in its *current* mode, without touching grads."""

    x, y = data_batch[0], data_batch[1]
    device = next(model.parameters()).device
    return float(criterion(model(x.to(device)), y.to(device)).detach().cpu())
