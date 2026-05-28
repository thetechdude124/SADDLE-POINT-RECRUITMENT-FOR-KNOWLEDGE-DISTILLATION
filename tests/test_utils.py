"""Tests for the utils module."""

import torch

from sprkd.utils import get_device, set_seed, to_device


def test_get_device_default_returns_torch_device():
    d = get_device()
    assert isinstance(d, torch.device)


def test_get_device_explicit_cpu():
    d = get_device("cpu")
    assert d.type == "cpu"


def test_get_device_unknown_falls_back_to_cpu_or_available():
    d = get_device("not-a-real-backend")
    assert isinstance(d, torch.device)


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.rand(3)
    set_seed(123)
    b = torch.rand(3)
    assert torch.allclose(a, b)


def test_to_device_recursive():
    cpu = torch.device("cpu")
    obj = {
        "a": torch.zeros(3),
        "b": [torch.ones(2), torch.ones(2)],
        "c": (torch.zeros(1),),
        "d": "string",
    }
    out = to_device(obj, cpu)
    assert out["a"].device.type == "cpu"
    for t in out["b"]:
        assert t.device.type == "cpu"
    assert out["c"][0].device.type == "cpu"
    assert out["d"] == "string"
