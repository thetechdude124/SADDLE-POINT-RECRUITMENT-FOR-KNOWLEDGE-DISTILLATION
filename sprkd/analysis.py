"""Hessian-spectrum analysis utilities used in Section 5 of the paper.

Wraps the PyHessian density / trace estimators with a small, dependency-light
interface so the rest of the package and downstream notebooks need not pin a
specific PyHessian version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class HessianSpectrum:
    """Aggregated Hessian-eigenspectrum statistics for a single model."""

    eigenvalues: list
    weights: list
    trace: float
    top_eigenvalue: float
    spectral_radius: float

    def to_dict(self) -> dict:
        return {
            "eigenvalues": self.eigenvalues,
            "weights": self.weights,
            "trace": self.trace,
            "top_eigenvalue": self.top_eigenvalue,
            "spectral_radius": self.spectral_radius,
        }


def compute_spectrum(
    model: nn.Module,
    criterion: nn.Module,
    data: tuple,
    *,
    n_iter: int = 100,
    n_v: int = 1,
    use_cuda: Optional[bool] = None,
) -> HessianSpectrum:
    """Compute Hessian density / trace / top eigenvalue via PyHessian."""

    from pyhessian import hessian as PyHessian

    if use_cuda is None:
        use_cuda = next(model.parameters()).is_cuda

    hess = PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)
    eigenvalues, weights = hess.density(iter=n_iter, n_v=n_v)
    trace = float(np.mean(hess.trace()))
    top_eigs, _ = hess.eigenvalues(top_n=1)
    top = float(top_eigs[0]) if len(top_eigs) > 0 else 0.0
    return HessianSpectrum(
        eigenvalues=eigenvalues,
        weights=weights,
        trace=trace,
        top_eigenvalue=top,
        spectral_radius=abs(top),
    )


def density_to_histogram(
    eigenvalues: list,
    weights: list,
    bins: int = 200,
    sigma_squared: float = 1e-5,
    eps: float = 1e-12,
    *,
    range_: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert PyHessian density samples to a (centers, density) histogram.

    Uses Gaussian smoothing identical to ``pyhessian.density_plot.get_esd_plot``
    but returns the raw arrays so callers can re-style the figure. Default
    ``sigma_squared`` matches the paper's ESD plots.
    """

    eigs = np.real(np.asarray(eigenvalues, dtype=np.float64).ravel())
    wts = np.real(np.asarray(weights, dtype=np.float64).ravel())
    if range_ is None:
        lo = float(eigs.min()) - 1.0
        hi = float(eigs.max()) + 1.0
    else:
        lo, hi = range_
    grid = np.linspace(lo, hi, bins)
    density = np.zeros_like(grid)
    for ev, w in zip(eigs, wts):
        density += w * np.exp(
            -((grid - ev) ** 2) / (2.0 * sigma_squared)
        ) / np.sqrt(2.0 * np.pi * sigma_squared)
    density = density / (np.sum(density) + eps)
    return grid, density


def hessian_trace(
    model: nn.Module,
    criterion: nn.Module,
    data: tuple,
    *,
    n_iter: int = 100,
    use_cuda: Optional[bool] = None,
) -> float:
    """Estimate Tr(H) by Hutchinson averaging (delegates to PyHessian)."""

    from pyhessian import hessian as PyHessian

    if use_cuda is None:
        use_cuda = next(model.parameters()).is_cuda
    hess = PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)
    return float(np.mean(hess.trace(maxIter=n_iter)))
