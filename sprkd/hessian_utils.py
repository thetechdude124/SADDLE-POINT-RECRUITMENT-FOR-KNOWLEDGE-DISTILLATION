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


# --------------------------------------------------------------------------- #
# Direct Hessian-vector products and extreme eigenpairs
# --------------------------------------------------------------------------- #

class HessianOperator:
    """Hessian-vector products of ``loss_fn(model(x), y)`` on one fixed batch.

    Builds the gradient graph once (``create_graph=True``) and evaluates ``H v`` by a
    second backward pass. Vectors are flat 1-D tensors over the trainable parameters.
    ``n_hvp`` counts the products performed.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, batch: tuple):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.shapes = [p.shape for p in self.params]
        self.numels = [p.numel() for p in self.params]
        self.n = sum(self.numels)
        self.device = self.params[0].device
        x, y = batch[0].to(self.device), batch[1].to(self.device)
        self.loss = loss_fn(model(x), y)
        self.grads = torch.autograd.grad(self.loss, self.params, create_graph=True)
        self.grad_flat = torch.cat([g.reshape(-1) for g in self.grads]).detach()
        self.n_hvp = 0

    def unflatten(self, v: torch.Tensor):
        out, i = [], 0
        for shp, n in zip(self.shapes, self.numels):
            out.append(v[i:i + n].view(shp)); i += n
        return out

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        vs = self.unflatten(v.to(self.device, dtype=self.params[0].dtype))
        hv = torch.autograd.grad(self.grads, self.params, grad_outputs=vs, retain_graph=True, allow_unused=True)
        self.n_hvp += 1
        return torch.cat([(h if h is not None else torch.zeros_like(p)).reshape(-1) for h, p in zip(hv, self.params)]).detach()

    def release(self):
        self.grads = None
        self.loss = None


def _power_iteration(matvec, n, device, dtype, max_iter=100, tol=1e-3, v0=None):
    v = v0 if v0 is not None else torch.randn(n, device=device, dtype=dtype)
    v = v / v.norm()
    lam = 0.0
    for _ in range(max_iter):
        hv = matvec(v)
        lam_new = float(torch.dot(v, hv))
        nrm = float(hv.norm())
        if nrm == 0.0:
            return 0.0, v
        v = hv / nrm
        if abs(lam_new - lam) <= tol * max(abs(lam_new), 1e-12):
            lam = lam_new
            break
        lam = lam_new
    return lam, v


def extreme_eigenpairs(
    model: nn.Module,
    loss_fn: nn.Module,
    batch: tuple,
    k: int = 1,
    *,
    method: str = "lanczos",
    max_iter: int = 100,
    tol: float = 1e-3,
) -> dict:
    """Largest and most negative Hessian eigenpairs on ``batch``.

    ``method="lanczos"`` uses ``scipy.sparse.linalg.eigsh`` (implicitly restarted Lanczos)
    on a matrix-free operator, ``which="LA"`` then ``which="SA"``. ``method="power"`` uses
    power iteration for the dominant eigenpair and shifted power iteration on
    ``H - lambda_dom I`` for the opposite extreme. Only ``k=1`` is supported. The model's
    train/eval mode and gradients are preserved; on MPS the computation runs on CPU.

    Returns ``{"lambda_max", "v_max", "lambda_min", "v_min", "n_hvp", "method", "grad_norm"}``
    with eigenvectors as lists of per-parameter tensors (PyHessian layout).
    """

    if k < 1:
        raise ValueError("k must be >= 1")
    if k > 1 and method != "lanczos":
        raise NotImplementedError("k > 1 is supported with method='lanczos' only")
    with hessian_compatible(model, batch) as (m, b, _):
        m.eval()  # deterministic loss on the probe batch (no dropout); mode is restored on exit
        op = HessianOperator(m, loss_fn, b)
        n, dev, dt = op.n, op.device, op.params[0].dtype
        used = method
        if method == "lanczos":
            try:
                import numpy as np
                from scipy.sparse.linalg import LinearOperator, eigsh

                def mv(x):
                    return op.hvp(torch.as_tensor(np.asarray(x, dtype=np.float64).ravel(), dtype=dt, device=dev)).double().cpu().numpy()

                lin = LinearOperator((n, n), matvec=mv, dtype=np.float64)
                ncv = min(n, max(20, 4 * k + 1))
                w_max, v_max = eigsh(lin, k=1, which="LA", tol=tol, maxiter=max_iter * 10, ncv=ncv)
                w_min, v_min = eigsh(lin, k=k, which="SA", tol=tol, maxiter=max_iter * 10, ncv=ncv)
                order = np.argsort(w_min)
                w_min, v_min = w_min[order], v_min[:, order]
                lam_max, lam_min = float(w_max[0]), float(w_min[0])
                lambda_min_k = [float(w) for w in w_min]
                vmax = torch.as_tensor(v_max[:, 0], dtype=dt, device=dev)
                vmin = torch.as_tensor(v_min[:, 0], dtype=dt, device=dev)
            except Exception:  # scipy missing or Lanczos failure: fall back to power iteration
                used = "power"
        if used == "power":
            lam_dom, v_dom = _power_iteration(op.hvp, n, dev, dt, max_iter, tol)
            lam_other, v_other = _power_iteration(lambda v: op.hvp(v) - lam_dom * v, n, dev, dt, max_iter, tol)
            lam_other = lam_other + lam_dom
            if lam_dom >= lam_other:
                lam_max, vmax, lam_min, vmin = lam_dom, v_dom, lam_other, v_other
            else:
                lam_max, vmax, lam_min, vmin = lam_other, v_other, lam_dom, v_dom
            lambda_min_k = [lam_min]
        grad_norm = float(op.grad_flat.norm())
        n_hvp = op.n_hvp
        out = {
            "lambda_max": lam_max, "lambda_min": lam_min, "lambda_min_k": lambda_min_k,
            "n_hvp": n_hvp, "method": used, "grad_norm": grad_norm,
            "v_max": [t.detach().clone() for t in op.unflatten(vmax / vmax.norm())],
            "v_min": [t.detach().clone() for t in op.unflatten(vmin / vmin.norm())],
        }
        op.release()
    return out
