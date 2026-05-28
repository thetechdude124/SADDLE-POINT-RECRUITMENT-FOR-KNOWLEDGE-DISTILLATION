"""Tests for sprkd.eval - trial-averaged evaluation utilities."""

import pytest
import torch
import torch.nn as nn

from sprkd.eval import (
    TrialResult,
    collect_predictions,
    evaluate_on_testset,
    evaluate_performance_trials,
)
from sprkd.models import MalariaStudentCNN


class _SingleSampleLoader:
    """Mimic the canonical Colab's ``test_loader`` (batch_size=1)."""

    def __init__(self, n: int, x_shape=(3, 32, 32)):
        torch.manual_seed(0)
        self.xs = torch.randn(n, *x_shape)
        self.ys = torch.randint(0, 2, (n,))
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield self.xs[i : i + 1], self.ys[i : i + 1]

    def __len__(self):
        return self.n


def test_trial_result_avg_accuracy_and_loss():
    res = TrialResult(model="m", accuracies=[80.0, 90.0, 100.0], losses=[0.1, 0.2, 0.3])
    assert res.avg_accuracy == pytest.approx(90.0)
    assert res.avg_loss == pytest.approx(0.2)


def test_trial_result_avg_handles_empty():
    res = TrialResult(model="m", accuracies=[], losses=[])
    assert res.avg_accuracy != res.avg_accuracy  # NaN
    assert res.avg_loss != res.avg_loss


def test_evaluate_on_testset_basic_smoke():
    model = MalariaStudentCNN()
    loader = _SingleSampleLoader(n=10)
    acc, loss = evaluate_on_testset(model, loader, n_samples=10, device=torch.device("cpu"))
    assert 0 <= acc <= 100
    assert loss >= 0


def test_evaluate_on_testset_handles_zero_samples():
    model = MalariaStudentCNN()
    loader = _SingleSampleLoader(n=10)
    acc, loss = evaluate_on_testset(model, loader, n_samples=0, device=torch.device("cpu"))
    assert acc != acc and loss != loss  # both NaN


def test_evaluate_performance_trials_returns_per_model_results():
    models = {
        "A": MalariaStudentCNN(),
        "B": MalariaStudentCNN(),
    }
    loader = _SingleSampleLoader(n=20)
    results = evaluate_performance_trials(
        models,
        loader,
        n_samples=8,
        n_trials=3,
        device=torch.device("cpu"),
    )
    assert len(results) == 2
    for r in results:
        assert len(r.accuracies) == 3
        assert len(r.losses) == 3


def test_collect_predictions_returns_aligned_tensors():
    model = MalariaStudentCNN()
    loader = _SingleSampleLoader(n=12)
    preds, targets = collect_predictions(model, loader, n_samples=10, device=torch.device("cpu"))
    assert preds.shape == targets.shape == (10,)
    assert preds.dtype == torch.long


def test_collect_predictions_empty_loader():
    model = MalariaStudentCNN()
    loader = _SingleSampleLoader(n=0)
    preds, targets = collect_predictions(model, loader, device=torch.device("cpu"))
    assert preds.numel() == 0 and targets.numel() == 0
