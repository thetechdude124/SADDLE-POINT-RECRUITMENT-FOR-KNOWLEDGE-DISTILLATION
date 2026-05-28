"""Tests for sprkd.visualize plotting helpers."""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI

import matplotlib.pyplot as plt  # noqa: E402  (must be after backend selection)

from sprkd.visualize import compare_esds, plot_esd, plot_loss_accuracy


def test_plot_loss_accuracy_returns_figure():
    histories = {
        "SPRKD":   {"valid_losses": [0.5, 0.4, 0.3], "valid_accuracies": [70, 80, 90]},
        "Control": {"valid_losses": [0.6, 0.5, 0.4], "valid_accuracies": [65, 75, 85]},
    }
    fig = plot_loss_accuracy(histories)
    assert fig is not None
    plt.close(fig)


def test_plot_esd_returns_axes():
    eigs = [-1.0, 0.0, 0.5, 1.0]
    wts = [0.1, 0.4, 0.3, 0.2]
    fig, ax = plot_esd(eigs, wts, log_y=False)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_compare_esds_handles_multiple():
    spec_a = {"eigenvalues": [-1.0, 0.0, 1.0], "weights": [0.3, 0.4, 0.3]}
    spec_b = {"eigenvalues": [-2.0, 0.0, 2.0], "weights": [0.5, 0.0, 0.5]}
    fig = compare_esds({"A": spec_a, "B": spec_b})
    assert fig is not None
    plt.close(fig)


def test_compare_esds_single_entry():
    spec = {"eigenvalues": [-1.0, 1.0], "weights": [0.5, 0.5]}
    fig = compare_esds({"only": spec})
    assert fig is not None
    plt.close(fig)


def test_plot_loss_landscape_from_dict():
    """Smoke-test the Figure-4 helper using a synthetic landscape."""

    import numpy as np

    from sprkd.visualize import plot_loss_landscape

    grid = 11
    alphas = np.linspace(-1, 1, grid)
    betas = np.linspace(-1, 1, grid)
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    losses = (A**2 + B**2)  # convex bowl

    fig = plot_loss_landscape(
        {"alphas": alphas, "betas": betas, "losses": losses},
        title="synthetic bowl",
    )
    assert fig is not None
    plt.close(fig)


def test_plot_loss_landscape_from_landscape_object():
    import numpy as np

    from sprkd.landscape import LossLandscape
    from sprkd.visualize import plot_loss_landscape

    grid = 7
    landscape = LossLandscape(
        alphas=np.linspace(-2, 2, grid),
        betas=np.linspace(-2, 2, grid),
        losses=np.random.rand(grid, grid),
        eigenvalues=[2.0, 1.0],
    )
    fig = plot_loss_landscape(landscape)
    assert fig is not None
    plt.close(fig)
