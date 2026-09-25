"""High-level training and evaluation loops for the three SPRKD model roles.

Each helper returns a :class:`TrainingHistory` object with per-step training
and per-epoch validation metrics. The functions are deliberately kept
framework-agnostic (no ``fastai`` Learner objects) so the package works in any
PyTorch environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sprkd.optimizer import SPRKD


@dataclass
class TrainingHistory:
    train_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    valid_losses: List[float] = field(default_factory=list)
    valid_accuracies: List[float] = field(default_factory=list)

    def best_valid_acc(self) -> float:
        return max(self.valid_accuracies, default=float("nan"))

    def to_dict(self) -> dict:
        return {
            "TRAINING": {
                "LOSSES": self.train_losses,
                "ACCURACIES": self.train_accuracies,
            },
            "VALIDATION": {
                "LOSSES": self.valid_losses,
                "ACCURACIES": self.valid_accuracies,
            },
        }


# ---------------------------------------------------------------------------
# Internal: per-epoch loops
# ---------------------------------------------------------------------------

def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = torch.argmax(logits, dim=1)
    return (pred == targets).float().mean().item() * 100.0


def _epoch_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            total_acc += _accuracy(logits, y)
            n_batches += 1
    if n_batches == 0:
        return float("nan"), float("nan")
    return total_loss / n_batches, total_acc / n_batches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_teacher(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    n_epochs: int,
    lr: float = 1e-3,
    saddle_steps: int = 1,
    n_top_eigs: int = 4,
    device: Optional[torch.device] = None,
    progress: bool = True,
    sprkd_kwargs: Optional[dict] = None,
) -> tuple[SPRKD, TrainingHistory]:
    """Train a teacher with SPRKD saddle tracking enabled.

    Returns the configured ``SPRKD`` optimizer (with its
    ``saddle_repository`` populated) and a :class:`TrainingHistory`.
    """

    device = device or next(model.parameters()).device
    model.to(device)
    base = torch.optim.Adam(model.parameters(), lr=lr)
    sprkd = SPRKD(
        model.parameters(),
        base_optimizer=base,
        loss_fn=loss_fn,
        is_teacher=True,
        saddle_steps=saddle_steps,
        n_top_eigs=n_top_eigs,
        **(sprkd_kwargs or {}),
    )
    history = _train_loop(
        model=model,
        optimizer=sprkd,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        device=device,
        progress=progress,
        sprkd_aware=True,
    )
    return sprkd, history


def train_control(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    n_epochs: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> TrainingHistory:
    """Train a scratch baseline (no SPRKD logic) - equivalent to vanilla Adam."""

    device = device or next(model.parameters()).device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return _train_loop_simple(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        device=device,
        progress=progress,
    )


def train_student(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    teacher_saddle_points: Sequence[torch.Tensor],
    n_epochs: int,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    progress: bool = True,
    sprkd_kwargs: Optional[dict] = None,
) -> tuple[SPRKD, TrainingHistory]:
    """Train a student with the full SPRKD pipeline (TM -> NHE -> PGD)."""

    device = device or next(model.parameters()).device
    model.to(device)
    base = torch.optim.Adam(model.parameters(), lr=lr)
    sprkd = SPRKD(
        model.parameters(),
        base_optimizer=base,
        loss_fn=loss_fn,
        is_teacher=False,
        is_control=False,
        teacher_saddle_points=[t.to(device) for t in teacher_saddle_points],
        saddle_steps=None,
        **(sprkd_kwargs or {}),
    )
    history = _train_loop(
        model=model,
        optimizer=sprkd,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        device=device,
        progress=progress,
        sprkd_aware=True,
    )
    return sprkd, history


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.5,
    temperature: float = 4.0,
) -> torch.Tensor:
    """Standard Hinton (2015) distillation loss on logits.

    ``alpha * CE(student, y) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))``.
    The ``T^2`` factor keeps the soft-target gradient magnitude comparable across
    temperatures. CRD (Tian et al. 2020) uses ``alpha = 0.1, T = 4`` on CIFAR-100; the
    default here is ``alpha = 0.5``.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    ce = nn.functional.cross_entropy(student_logits, targets)
    log_p_s = torch.log_softmax(student_logits / temperature, dim=1)
    p_t = torch.softmax(teacher_logits / temperature, dim=1)
    kl = nn.functional.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)
    return alpha * ce + (1.0 - alpha) * kl


def train_response_kd(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    n_epochs: int,
    lr: float = 1e-3,
    temperature: float = 4.0,
    alpha: float = 0.5,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> TrainingHistory:
    """Response KD baseline (Hinton et al. 2015) with the standard combined loss.

    Same optimizer (Adam, ``lr``) and epoch budget as the other arms. ``alpha`` is
    the weight on the cross-entropy term, ``1 - alpha`` on the ``T^2``-scaled KL
    term; see :func:`kd_loss`. Validation loss and accuracy are computed with plain
    cross-entropy on the task labels so they are comparable across arms.
    """

    device = device or next(student.parameters()).device
    student.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    history = TrainingHistory()
    iterator = range(n_epochs)
    if progress:
        iterator = tqdm(iterator, desc="rkd-train")

    for epoch in iterator:
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, y, alpha=alpha, temperature=temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.train_losses.append(loss.item())
            history.train_accuracies.append(_accuracy(s_logits, y))
        v_loss, v_acc = _epoch_eval(
            student, valid_loader, nn.CrossEntropyLoss(), device
        )
        history.valid_losses.append(v_loss)
        history.valid_accuracies.append(v_acc)
    return history


# ---------------------------------------------------------------------------
# Internal training loops
# ---------------------------------------------------------------------------

def _train_loop(
    *,
    model: nn.Module,
    optimizer: SPRKD,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    n_epochs: int,
    device: torch.device,
    progress: bool,
    sprkd_aware: bool,
) -> TrainingHistory:
    history = TrainingHistory()
    model._steps_per_epoch = len(train_loader)  # used by PGD epoch limit

    epoch_iter: Iterable[int] = range(n_epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="train")

    for _ in epoch_iter:
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if sprkd_aware:
                optimizer.step(
                    model=model,
                    current_loss=loss.detach(),
                    data_batch=(x, y),
                )
            else:
                optimizer.step()
            history.train_losses.append(loss.item())
            history.train_accuracies.append(_accuracy(logits, y))

        v_loss, v_acc = _epoch_eval(model, valid_loader, loss_fn, device)
        history.valid_losses.append(v_loss)
        history.valid_accuracies.append(v_acc)
    return history


def _train_loop_simple(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    n_epochs: int,
    device: torch.device,
    progress: bool,
) -> TrainingHistory:
    history = TrainingHistory()
    epoch_iter: Iterable[int] = range(n_epochs)
    if progress:
        epoch_iter = tqdm(epoch_iter, desc="control")

    for _ in epoch_iter:
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            history.train_losses.append(loss.item())
            history.train_accuracies.append(_accuracy(logits, y))
        v_loss, v_acc = _epoch_eval(model, valid_loader, loss_fn, device)
        history.valid_losses.append(v_loss)
        history.valid_accuracies.append(v_acc)
    return history
