"""2-D loss landscape visualization.

Reproduces Figure 4 of the SPRKD paper: the loss landscape near a converged
model, projected onto the top-2 Hessian eigenvectors and perturbed across
``[lambda_min, lambda_max]`` per axis.

The procedure:

1. Compute the top-2 Hessian eigenpairs of the model on a representative
   batch (delegates to PyHessian; MPS is handled via the same compatibility
   shim used by :class:`sprkd.optimizer.SPRKD`).
2. Snapshot the model's current parameters.
3. For each ``(alpha, beta)`` pair on a regular grid, set the model
   parameters to ``theta + alpha * v_1 + beta * v_2`` and evaluate the loss
   on the same batch.
4. Restore the original parameters.

The resulting ``(alphas, betas, losses)`` arrays can be passed to
:func:`sprkd.visualize.plot_loss_landscape` to render the 3-D surface and
2-D heatmap in the paper.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LossLandscape:
    """Container for the (alphas, betas, losses) tensor of a 2-D landscape sweep."""

    alphas: np.ndarray
    betas: np.ndarray
    losses: np.ndarray
    eigenvalues: List[float]

    def to_dict(self) -> dict:
        return {
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
            "losses": self.losses.tolist(),
            "eigenvalues": list(self.eigenvalues),
        }


def _flatten(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


@contextmanager
def _temp_parameters(model: nn.Module, new_flat: torch.Tensor):
    """Temporarily set ``model`` parameters from a flat 1-D tensor."""

    old = [p.detach().clone() for p in model.parameters()]
    try:
        offset = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(new_flat[offset : offset + n].reshape_as(p))
                offset += n
        yield model
    finally:
        with torch.no_grad():
            for p, snap in zip(model.parameters(), old):
                p.copy_(snap)


def _to_pyhessian_compat(
    model: nn.Module, data: tuple
) -> tuple[nn.Module, tuple, bool, torch.device]:
    """Return (model, data, use_cuda, original_device).

    PyHessian only knows ``cuda`` and ``cpu``; on MPS we move both model
    and data to CPU (the caller is expected to migrate them back).
    """

    original = next(model.parameters()).device
    if original.type == "mps":
        model.to("cpu")
        if isinstance(data, (tuple, list)):
            data = tuple(d.to("cpu") if hasattr(d, "to") else d for d in data)
        return model, data, False, original
    return model, data, original.type == "cuda", original


def compute_top2_eigenpairs(
    model: nn.Module,
    criterion: nn.Module,
    data: tuple,
) -> Tuple[List[float], List[List[torch.Tensor]]]:
    """Top-2 Hessian eigenvalues + eigenvectors via PyHessian.

    ``eigenvectors[k]`` is a list-of-tensors with the same shape as
    ``model.parameters()``.
    """

    from pyhessian import hessian as PyHessian

    model, data, use_cuda, original = _to_pyhessian_compat(model, data)
    try:
        hess = PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)
        eigs, vecs = hess.eigenvalues(top_n=2)
    finally:
        if next(model.parameters()).device != original:
            model.to(original)
    return [float(e) for e in eigs], vecs


def compute_landscape(
    model: nn.Module,
    criterion: nn.Module,
    data: tuple,
    *,
    grid_size: int = 21,
    range_alpha: Optional[Tuple[float, float]] = None,
    range_beta: Optional[Tuple[float, float]] = None,
    eigenvalues: Optional[List[float]] = None,
    eigenvectors: Optional[List[List[torch.Tensor]]] = None,
    progress: bool = False,
) -> LossLandscape:
    """Sweep loss over a 2-D grid of perturbations along the top-2 eigenvectors.

    Parameters
    ----------
    model, criterion, data
        Standard supervised triple. ``data`` should be a small batch.
    grid_size : int
        Number of points per axis (``grid_size x grid_size`` evaluations).
    range_alpha, range_beta : (float, float), optional
        Coefficient ranges for the two eigendirections. By default both
        ranges are scaled by the corresponding eigenvalue magnitude as in the
        paper (``[-|lambda|, +|lambda|]``).
    eigenvalues, eigenvectors : optional
        Pre-computed eigenpairs. If ``None`` the function calls
        :func:`compute_top2_eigenpairs` internally.

    Returns
    -------
    LossLandscape
    """

    if eigenvalues is None or eigenvectors is None:
        eigenvalues, eigenvectors = compute_top2_eigenpairs(model, criterion, data)

    if len(eigenvalues) < 2 or len(eigenvectors) < 2:
        raise ValueError(
            f"Need top-2 eigenpairs; got {len(eigenvalues)} eigenvalues "
            f"and {len(eigenvectors)} eigenvectors."
        )

    if range_alpha is None:
        s = abs(eigenvalues[0]) or 1.0
        range_alpha = (-s, s)
    if range_beta is None:
        s = abs(eigenvalues[1]) or 1.0
        range_beta = (-s, s)

    alphas = np.linspace(range_alpha[0], range_alpha[1], grid_size)
    betas = np.linspace(range_beta[0], range_beta[1], grid_size)
    losses = np.zeros((grid_size, grid_size), dtype=np.float64)

    device = next(model.parameters()).device
    v1 = _flatten(eigenvectors[0]).to(device)
    v2 = _flatten(eigenvectors[1]).to(device)
    theta0 = _flatten(model.parameters()).to(device)

    if isinstance(data, (tuple, list)):
        x, y = data
        x, y = x.to(device), y.to(device)
    else:
        raise ValueError("`data` must be a (inputs, targets) tuple.")

    iterator = range(grid_size)
    if progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="landscape")

    model_was_training = model.training
    model.eval()
    try:
        for i in iterator:
            for j in range(grid_size):
                shifted = theta0 + alphas[i] * v1 + betas[j] * v2
                with _temp_parameters(model, shifted):
                    with torch.no_grad():
                        logits = model(x)
                        losses[i, j] = float(criterion(logits, y).item())
    finally:
        if model_was_training:
            model.train()

    return LossLandscape(
        alphas=alphas,
        betas=betas,
        losses=losses,
        eigenvalues=list(eigenvalues),
    )
