"""Trial-averaged model evaluation, mirroring the canonical Colab notebook.

The TinyImageNet evaluation in
``EXPERIMENTAL_MODEL_EVALUATIONS.ipynb`` averages per-trial accuracy and
loss across N independent slices of the validation loader. This module
exposes the same logic as a tidy public API so users can reproduce the
paper's reported numbers from any saved checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


@dataclass
class TrialResult:
    """Per-model averaged metrics across ``n_trials`` trials."""

    model: str
    accuracies: List[float]
    losses: List[float]

    @property
    def avg_accuracy(self) -> float:
        return mean(self.accuracies) if self.accuracies else float("nan")

    @property
    def avg_loss(self) -> float:
        return mean(self.losses) if self.losses else float("nan")

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "accuracies": list(self.accuracies),
            "losses": list(self.losses),
            "avg_accuracy": self.avg_accuracy,
            "avg_loss": self.avg_loss,
        }


def evaluate_on_testset(
    model: nn.Module,
    test_loader,
    n_samples: int,
    *,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> tuple[float, float]:
    """Evaluate ``model`` on the first ``n_samples`` of ``test_loader``.

    Mirrors :func:`evaluate_on_testset` in
    ``EXPERIMENTAL_MODEL_EVALUATIONS.ipynb``: assumes the loader emits
    one sample per batch (``batch_size=1``) so ``n_samples`` controls the
    sample count exactly.
    """

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    total_loss = 0.0
    seen = 0
    with torch.no_grad():
        for x, y in islice(iter(test_loader), n_samples):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            seen += y.numel()
    if seen == 0:
        return float("nan"), float("nan")
    return 100.0 * correct / seen, total_loss / seen


def evaluate_performance_trials(
    models: Dict[str, nn.Module],
    test_loader,
    *,
    n_samples: int,
    n_trials: int,
    loss_fn: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> List[TrialResult]:
    """Replicate ``evaluate_performance_trials`` from the canonical Colab.

    Each trial draws a fresh ``n_samples``-prefix from ``test_loader``; the
    final result is the mean accuracy / loss across all trials per model.

    Parameters
    ----------
    models : dict
        ``{name: nn.Module}``. Names appear in the returned :class:`TrialResult`s.
    test_loader : torch.utils.data.DataLoader
        Validation/test loader. Should be shuffled so successive trials see
        different prefixes (canonical Colab uses ``shuffle=True, batch_size=1``).
    n_samples : int
        Samples per trial.
    n_trials : int
        Number of independent trials to average over.
    """

    results: Dict[str, TrialResult] = {
        name: TrialResult(model=name, accuracies=[], losses=[]) for name in models
    }
    for trial in range(n_trials):
        for name, m in models.items():
            acc, loss = evaluate_on_testset(
                m, test_loader, n_samples=n_samples, loss_fn=loss_fn, device=device
            )
            results[name].accuracies.append(acc)
            results[name].losses.append(loss)
            if verbose:
                print(
                    f"[trial {trial + 1}/{n_trials}] {name:20s} "
                    f"acc={acc:.4f} loss={loss:.6f}"
                )
    return list(results.values())


def collect_predictions(
    model: nn.Module,
    test_loader,
    *,
    n_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference and return ``(preds, targets)`` for downstream stats.

    Useful as the input to :func:`sprkd.stats.mcnemar_test`.
    """

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    preds_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    iterator = iter(test_loader) if n_samples is None else islice(iter(test_loader), n_samples)
    with torch.no_grad():
        for x, y in iterator:
            x = x.to(device)
            logits = model(x)
            preds_list.append(torch.argmax(logits, dim=1).cpu())
            targets_list.append(y.cpu())
    if not preds_list:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(preds_list), torch.cat(targets_list)
