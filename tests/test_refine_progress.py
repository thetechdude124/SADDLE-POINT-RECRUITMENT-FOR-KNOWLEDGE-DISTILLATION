"""refine_to_stationary reports progress through progress_cb (added after the 2026-09-28
Modal overrun, where refinement runs printed nothing for hours)."""

import torch
import torch.nn as nn

from sprkd.saddle import refine_to_stationary


def _toy():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))
    x = torch.randn(16, 4); y = torch.randint(0, 3, (16,))
    return m, (x, y)


def test_progress_cb_called_every_step_with_increasing_elapsed():
    m, batch = _toy()
    calls = []
    rec = refine_to_stationary(m, nn.CrossEntropyLoss(), batch, max_steps=5, grad_tol=1e-9, lr=1e-3,
                               method="adam", progress_cb=lambda step, gn, el: calls.append((step, gn, el)))
    assert rec["steps"] == 5
    steps = [c[0] for c in calls]
    assert steps[-1] == 5 and steps == sorted(steps)
    assert all(c[2] >= 0 for c in calls) and calls[-1][2] >= calls[0][2]
    assert all(isinstance(c[1], float) for c in calls)


def test_progress_cb_optional():
    m, batch = _toy()
    rec = refine_to_stationary(m, nn.CrossEntropyLoss(), batch, max_steps=2, method="adam", lr=1e-3)
    assert rec["steps"] == 2
