"""Tests for sprkd.landscape - 2-D loss-landscape sweeps."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from sprkd.landscape import (
    LossLandscape,
    _flatten,
    _temp_parameters,
    compute_landscape,
)
from sprkd.models import MalariaStudentCNN


def test_flatten_concatenates_in_order():
    a = torch.arange(6.0).reshape(2, 3)
    b = torch.arange(6.0, 10.0).reshape(2, 2)
    flat = _flatten([a, b])
    assert flat.shape == (10,)
    assert torch.allclose(flat[:6], a.reshape(-1))
    assert torch.allclose(flat[6:], b.reshape(-1))


def test_temp_parameters_round_trip():
    model = MalariaStudentCNN()
    snapshot = [p.detach().clone() for p in model.parameters()]
    flat = _flatten(model.parameters())
    perturbed = flat + 1.0

    with _temp_parameters(model, perturbed):
        # parameters must reflect the perturbed values
        for p, expected in zip(model.parameters(), snapshot):
            assert not torch.allclose(p.detach(), expected)
    # outside the context, parameters are restored exactly
    for p, expected in zip(model.parameters(), snapshot):
        assert torch.allclose(p.detach(), expected)


def test_compute_landscape_with_supplied_eigenpairs():
    """Avoid the Hessian computation by providing fake eigenpairs."""

    model = MalariaStudentCNN()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 2, (4,))

    fake_v1 = [torch.randn_like(p) for p in model.parameters()]
    fake_v2 = [torch.randn_like(p) for p in model.parameters()]
    fake_eigs = [1.0, 0.5]

    landscape = compute_landscape(
        model,
        criterion,
        (x, y),
        grid_size=5,
        eigenvalues=fake_eigs,
        eigenvectors=[fake_v1, fake_v2],
        progress=False,
    )
    assert isinstance(landscape, LossLandscape)
    assert landscape.alphas.shape == (5,)
    assert landscape.betas.shape == (5,)
    assert landscape.losses.shape == (5, 5)
    assert np.all(np.isfinite(landscape.losses))


def test_compute_landscape_default_ranges_use_eigenvalue_magnitudes():
    model = MalariaStudentCNN()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 2, (4,))
    v1 = [torch.zeros_like(p) for p in model.parameters()]
    v2 = [torch.zeros_like(p) for p in model.parameters()]
    landscape = compute_landscape(
        model, criterion, (x, y),
        grid_size=3,
        eigenvalues=[2.5, 0.7],
        eigenvectors=[v1, v2],
    )
    assert landscape.alphas[0] == pytest.approx(-2.5)
    assert landscape.alphas[-1] == pytest.approx(2.5)
    assert landscape.betas[0] == pytest.approx(-0.7)
    assert landscape.betas[-1] == pytest.approx(0.7)


def test_compute_landscape_restores_parameters():
    model = MalariaStudentCNN()
    snapshot = [p.detach().clone() for p in model.parameters()]
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 2, (4,))
    v1 = [torch.randn_like(p) for p in model.parameters()]
    v2 = [torch.randn_like(p) for p in model.parameters()]
    compute_landscape(
        model, criterion, (x, y),
        grid_size=4,
        eigenvalues=[1.0, 1.0],
        eigenvectors=[v1, v2],
    )
    for p, snap in zip(model.parameters(), snapshot):
        assert torch.allclose(p.detach(), snap)


def test_compute_landscape_validates_eigenpair_count():
    model = MalariaStudentCNN()
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 2, (4,))
    v1 = [torch.zeros_like(p) for p in model.parameters()]
    with pytest.raises(ValueError):
        compute_landscape(
            model, criterion, (x, y),
            grid_size=3,
            eigenvalues=[1.0],
            eigenvectors=[v1],
        )


def test_landscape_to_dict_serialisable():
    landscape = LossLandscape(
        alphas=np.linspace(-1, 1, 3),
        betas=np.linspace(-1, 1, 3),
        losses=np.zeros((3, 3)),
        eigenvalues=[1.0, 0.5],
    )
    d = landscape.to_dict()
    assert d["alphas"] == [-1.0, 0.0, 1.0]
    assert d["losses"] == [[0.0] * 3] * 3
    assert d["eigenvalues"] == [1.0, 0.5]
