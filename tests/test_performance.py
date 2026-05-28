"""Lightweight performance smoke tests (not formal benchmarks).

Guards against accidental regressions that make core operations unusably slow
on CPU/MPS. Full 500-epoch training is intentionally out of scope.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sprkd.landscape import compute_landscape
from sprkd.legacy import load_legacy_metrics_pkl
from sprkd.models import MalariaStudentCNN
from sprkd.optimizer import SPRKD


pytestmark = pytest.mark.performance

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_student_forward_pass_under_50ms(student_model):
    x = torch.randn(64, 3, 32, 32)
    student_model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            student_model(x)
    elapsed = (time.perf_counter() - t0) / 20
    assert elapsed < 0.05, f"mean forward pass {elapsed*1e3:.1f} ms exceeds 50 ms budget"


def test_control_optimizer_step_under_200ms(student_model, tiny_batch, cpu_loss):
    x, y = tiny_batch
    base = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    opt = SPRKD(
        student_model.parameters(),
        base_optimizer=base,
        loss_fn=cpu_loss,
        is_control=True,
    )
    logits = student_model(x)
    loss = cpu_loss(logits, y)
    loss.backward()
    t0 = time.perf_counter()
    opt.step()
    assert time.perf_counter() - t0 < 0.2


def test_small_landscape_grid_completes_under_30s(student_model, tiny_batch):
    model = student_model
    criterion = nn.CrossEntropyLoss()
    t0 = time.perf_counter()
    landscape = compute_landscape(
        model,
        criterion,
        tiny_batch,
        grid_size=7,
        progress=False,
    )
    elapsed = time.perf_counter() - t0
    assert landscape.losses.shape == (7, 7)
    assert elapsed < 30.0, f"7x7 landscape took {elapsed:.1f}s"


@pytest.mark.skipif(
    not (REPO_ROOT / "METRICS" / "LOSSES AND ACCURACIES" / "500_SPRKD_LOSSES.pkl").is_file(),
    reason="metrics pickle not present",
)
def test_legacy_metrics_pkl_loads_under_30s():
    path = REPO_ROOT / "METRICS" / "LOSSES AND ACCURACIES" / "500_SPRKD_LOSSES.pkl"
    t0 = time.perf_counter()
    metrics = load_legacy_metrics_pkl(path)
    elapsed = time.perf_counter() - t0
    assert "VALIDATION" in metrics
    assert elapsed < 30.0, f"pickle load took {elapsed:.1f}s"
