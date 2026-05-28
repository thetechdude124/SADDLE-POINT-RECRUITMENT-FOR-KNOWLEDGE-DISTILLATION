"""Plotting helpers for the figures in Section 5 of the paper."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - matplotlib is a soft dep at import-time
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None  # type: ignore[assignment]


def _require_matplotlib():
    if plt is None:  # pragma: no cover - guarded by tests
        raise ImportError(
            "matplotlib is required for sprkd.visualize. "
            "Install with `pip install matplotlib`."
        )


def plot_loss_accuracy(
    histories: dict,
    *,
    save_path: Optional[str] = None,
    title_loss: str = "Validation loss",
    title_acc: str = "Validation accuracy (%)",
):
    """Plot validation loss / accuracy curves for a set of named runs.

    Parameters
    ----------
    histories : dict
        Mapping ``{label: TrainingHistory or dict-like with valid_losses /
        valid_accuracies}``.
    save_path : str, optional
        If provided, save the figure to this path.
    """

    _require_matplotlib()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    def _get(obj, field: str):
        if hasattr(obj, field):
            return getattr(obj, field)
        return obj[field]

    for label, hist in histories.items():
        v_losses = _get(hist, "valid_losses")
        v_accs = _get(hist, "valid_accuracies")
        ax_loss.plot(v_losses, label=label)
        ax_acc.plot(v_accs, label=label)
    for ax, ylabel, title in (
        (ax_loss, "loss", title_loss),
        (ax_acc, "accuracy (%)", title_acc),
    ):
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_esd(
    eigenvalues: Iterable[float],
    weights: Iterable[float],
    *,
    sigma_squared: float = 1e-5,
    bins: int = 200,
    range_: Optional[Tuple[float, float]] = None,
    label: Optional[str] = None,
    save_path: Optional[str] = None,
    log_y: bool = True,
):
    """Plot a single Hessian Eigenvalue Spectral Density (ESD).

    Mirrors :func:`pyhessian.density_plot.get_esd_plot` but exposes axes for
    downstream styling.
    """

    _require_matplotlib()
    from sprkd.analysis import density_to_histogram

    grid, density = density_to_histogram(
        list(eigenvalues),
        list(weights),
        bins=bins,
        sigma_squared=sigma_squared,
        range_=range_,
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, density, label=label or "ESD")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def compare_esds(
    spectra: dict,
    *,
    sigma_squared: float = 1e-5,
    bins: int = 200,
    save_path: Optional[str] = None,
):
    """Multi-row ESD comparison akin to Figure 3 of the paper.

    ``spectra`` should be ``{label: HessianSpectrum or dict}``.
    """

    _require_matplotlib()
    from sprkd.analysis import density_to_histogram

    fig, axes = plt.subplots(1, len(spectra), figsize=(5 * len(spectra), 4))
    if len(spectra) == 1:
        axes = [axes]

    def _get(obj, field: str):
        if hasattr(obj, field):
            return getattr(obj, field)
        return obj[field]

    for ax, (label, spec) in zip(axes, spectra.items()):
        ev = _get(spec, "eigenvalues")
        wt = _get(spec, "weights")
        grid, density = density_to_histogram(
            ev, wt, bins=bins, sigma_squared=sigma_squared
        )
        ax.plot(grid, density)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("density")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_landscape(
    landscape,
    *,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = "viridis",
):
    """Render the (3-D surface, 2-D heatmap) pair of Figure 4 in the paper.

    Parameters
    ----------
    landscape : LossLandscape or dict
        Output of :func:`sprkd.landscape.compute_landscape`, or any object /
        dict with ``alphas``, ``betas``, ``losses`` arrays of equal grid size.
    title : str, optional
        Title to prepend to both subplots.
    save_path : str, optional
        If supplied, save the figure to this path at 150 DPI.
    """

    _require_matplotlib()
    import numpy as np

    def _get(field: str):
        if hasattr(landscape, field):
            return getattr(landscape, field)
        return landscape[field]

    alphas = np.asarray(_get("alphas"))
    betas = np.asarray(_get("betas"))
    losses = np.asarray(_get("losses"))
    A, B = np.meshgrid(alphas, betas, indexing="ij")

    fig = plt.figure(figsize=(12, 5))
    ax_surf = fig.add_subplot(1, 2, 1, projection="3d")
    ax_surf.plot_surface(A, B, losses, cmap=cmap, edgecolor="none")
    ax_surf.set_xlabel(r"$\alpha$ (top eigendirection)")
    ax_surf.set_ylabel(r"$\beta$ (2nd eigendirection)")
    ax_surf.set_zlabel("loss")
    if title:
        ax_surf.set_title(f"{title} (3-D)")
    else:
        ax_surf.set_title("Loss landscape (3-D)")

    ax_heat = fig.add_subplot(1, 2, 2)
    im = ax_heat.imshow(
        losses.T,
        extent=(alphas.min(), alphas.max(), betas.min(), betas.max()),
        origin="lower",
        aspect="auto",
        cmap=cmap,
    )
    ax_heat.set_xlabel(r"$\alpha$")
    ax_heat.set_ylabel(r"$\beta$")
    if title:
        ax_heat.set_title(f"{title} (heatmap)")
    else:
        ax_heat.set_title("Loss landscape (heatmap)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="loss")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
