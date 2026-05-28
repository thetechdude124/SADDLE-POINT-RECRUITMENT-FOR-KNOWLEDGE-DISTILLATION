"""Tests for sprkd.analysis utilities."""

import numpy as np

from sprkd.analysis import HessianSpectrum, density_to_histogram


def test_hessian_spectrum_to_dict_roundtrip():
    spec = HessianSpectrum(
        eigenvalues=[1.0, -1.0],
        weights=[0.5, 0.5],
        trace=33.39,
        top_eigenvalue=2.0,
        spectral_radius=2.0,
    )
    d = spec.to_dict()
    assert d["trace"] == 33.39
    assert d["spectral_radius"] == 2.0


def test_density_to_histogram_shapes():
    eigs = [[-1.0, 0.0, 1.0]]
    wts = [[0.3, 0.4, 0.3]]
    grid, density = density_to_histogram(eigs, wts, bins=64)
    assert grid.shape == (64,)
    assert density.shape == (64,)
    assert np.all(density >= 0)


def test_density_to_histogram_normalises():
    eigs = [[-1.0, 0.0, 1.0]]
    wts = [[0.3, 0.4, 0.3]]
    _, density = density_to_histogram(eigs, wts, bins=200, sigma_squared=1e-3)
    assert np.isclose(density.sum(), 1.0, atol=1e-5)


def test_density_to_histogram_respects_range():
    eigs = [[-2.0, 2.0]]
    wts = [[0.5, 0.5]]
    grid, _ = density_to_histogram(eigs, wts, range_=(-5.0, 5.0), bins=16)
    assert grid.min() == -5.0
    assert grid.max() == 5.0
