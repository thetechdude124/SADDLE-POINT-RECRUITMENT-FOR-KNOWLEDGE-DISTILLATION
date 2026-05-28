"""Shared pytest fixtures for the SPRKD test suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.utils import set_seed


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _deterministic():
    set_seed(0)
    yield


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def teacher_model() -> nn.Module:
    return MalariaTeacherCNN()


@pytest.fixture
def student_model() -> nn.Module:
    return MalariaStudentCNN()


@pytest.fixture
def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 2, (8,))
    return x, y


@pytest.fixture
def tiny_loader(tiny_batch):
    """A 4-batch ``DataLoader``-like iterator for fast smoke-tests."""

    class _Loader:
        def __init__(self, x, y, n=4):
            self.x = x
            self.y = y
            self.n = n

        def __iter__(self):
            for _ in range(self.n):
                yield self.x.clone(), self.y.clone()

        def __len__(self):
            return self.n

    return _Loader(*tiny_batch)


@pytest.fixture
def cpu_loss():
    return nn.CrossEntropyLoss()
