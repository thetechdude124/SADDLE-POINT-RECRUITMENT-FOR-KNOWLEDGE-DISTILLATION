"""Saddle-point detection and Approximated Saddle Region (ASR) construction.

Detection rules (paper Section 3.1, Eq. 1). ``Lambda`` is the set of estimated top
Hessian eigenvalues at a training step:

* ``"magnitude"`` (default; the rule used by the canonical Colab notebook and the released
  checkpoints): ``|sum_{lambda_i < 0} lambda_i| >= beta`` with ``beta = 7``.
* ``"ratio"`` (the original ISEF 2023 notebook): ``|sum neg| >= alpha * sum pos`` with
  ``alpha = 0.4``.
* ``"both"`` (paper Equation 1 read literally): both conditions.

All three rules can additionally be gated on the gradient norm
(``SaddleCriterion.max_grad_norm``): a true saddle has a vanishing gradient, and without
the gate the rules fire on ordinary early-training iterates that merely have some
negative curvature. The gate is off by default (``None``) so that released behaviour is
reproducible; every detection is logged with its gradient norm so the gate can be chosen
after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence, Union

import torch


_RuleName = Literal["magnitude", "ratio", "both"]


@dataclass
class SaddleCriterion:
    """Hyperparameters for the saddle-point detection rule.

    Parameters
    ----------
    rule : {"magnitude", "ratio", "both"}, default ``"magnitude"``
    alpha : float
        Ratio threshold (paper's :math:`\\alpha`), used by ``"ratio"`` and ``"both"``.
    magnitude_threshold : float
        Absolute negative-eigenvalue mass threshold (paper's :math:`\\beta`).
    require_negative_eigenvalue : bool
        At least one strictly negative eigenvalue must be present.
    max_grad_norm : float or None
        If set, a step qualifies only when the gradient L2 norm is at most this value.
    """

    rule: _RuleName = "magnitude"
    alpha: float = 0.4
    magnitude_threshold: float = 7.0
    require_negative_eigenvalue: bool = True
    max_grad_norm: Optional[float] = None


def _split_signs(eigenvalues: Sequence[float]):
    pos, neg, zero = [], [], []
    for ev in eigenvalues:
        ev = float(ev)
        if ev > 0:
            pos.append(ev)
        elif ev < 0:
            neg.append(ev)
        else:
            zero.append(ev)
    return pos, neg, zero


def which_rules_fire(
    eigenvalues: Sequence[float],
    criterion: Optional[SaddleCriterion] = None,
    grad_norm: Optional[float] = None,
) -> dict:
    """Evaluate every condition and return a dict of booleans plus the decision.

    Keys: ``magnitude``, ``ratio``, ``grad_gate`` (True when the gate passes or is off),
    ``has_negative``, ``fired`` (the final decision for ``criterion.rule``).
    """

    if criterion is None:
        criterion = SaddleCriterion()

    pos, neg, _ = _split_signs(eigenvalues)
    pos_mass = sum(pos)
    neg_mass = abs(sum(neg))
    out = {
        "has_negative": bool(neg),
        "ratio": bool(neg_mass >= criterion.alpha * pos_mass),
        "magnitude": bool(neg_mass >= criterion.magnitude_threshold),
        "grad_gate": bool(
            criterion.max_grad_norm is None
            or grad_norm is None
            or grad_norm <= criterion.max_grad_norm
        ),
        "neg_mass": neg_mass,
        "pos_mass": pos_mass,
    }
    if criterion.require_negative_eigenvalue and not out["has_negative"]:
        fired = False
    elif criterion.rule == "magnitude":
        fired = out["magnitude"]
    elif criterion.rule == "ratio":
        fired = out["ratio"]
    elif criterion.rule == "both":
        fired = out["ratio"] and out["magnitude"]
    else:
        raise ValueError(f"Unknown saddle rule: {criterion.rule!r}")
    out["fired"] = bool(fired and out["grad_gate"])
    return out


def is_strong_saddle_point(
    eigenvalues: Sequence[float],
    criterion: Optional[SaddleCriterion] = None,
    grad_norm: Optional[float] = None,
) -> bool:
    """Return ``True`` iff ``eigenvalues`` (and optionally ``grad_norm``) qualify."""

    return which_rules_fire(eigenvalues, criterion, grad_norm)["fired"]


@dataclass
class SaddlePointRepository:
    """Snapshots of qualifying parameter states for one teacher.

    Each ``append`` stores a CPU clone of the parameters together with the loss, the
    gradient norm and the rule outcome at that step. With ``top_k`` set, only the ``top_k``
    lowest-loss snapshots are retained (the worst is dropped on overflow).
    """

    teacher_index: int
    top_k: Optional[int] = None
    snapshots: List[List[torch.Tensor]] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    rules: List[dict] = field(default_factory=list)
    n_checked: int = 0
    n_dropped: int = 0

    def append(
        self,
        params: Iterable[torch.nn.Parameter],
        loss: Optional[float] = None,
        *,
        grad_norm: Optional[float] = None,
        step: Optional[int] = None,
        rule: Optional[dict] = None,
    ) -> None:
        cpu_snap = [p.clone().detach().to("cpu") for p in params]
        self.snapshots.append(cpu_snap)
        self.losses.append(float(loss) if loss is not None else float("nan"))
        self.grad_norms.append(float(grad_norm) if grad_norm is not None else float("nan"))
        self.steps.append(int(step) if step is not None else -1)
        self.rules.append(dict(rule) if rule is not None else {})
        if self.top_k is not None and len(self.snapshots) > self.top_k:
            self._drop_worst()

    def _drop_worst(self) -> None:
        finite = [(i, l) for i, l in enumerate(self.losses) if l == l]
        idx = max(finite, key=lambda kv: kv[1])[0] if finite else 0
        for lst in (self.snapshots, self.losses, self.grad_norms, self.steps, self.rules):
            del lst[idx]
        self.n_dropped += 1

    def __len__(self) -> int:  # noqa: D401 - magic method
        return len(self.snapshots)

    @property
    def best_index(self) -> int:
        if not self.snapshots:
            raise IndexError("SaddlePointRepository is empty.")
        finite = [(i, l) for i, l in enumerate(self.losses) if l == l]  # NaN-safe
        if not finite:
            return len(self.snapshots) - 1
        return min(finite, key=lambda kv: kv[1])[0]

    @property
    def best(self) -> List[torch.Tensor]:
        """Lowest-loss snapshot, or the most recent one if losses are unset."""

        return self.snapshots[self.best_index]

    def summary(self) -> dict:
        return {
            "teacher_index": self.teacher_index,
            "n_checked": self.n_checked,
            "n_recorded": len(self.snapshots),
            "n_dropped": self.n_dropped,
            "losses": list(self.losses),
            "grad_norms": list(self.grad_norms),
            "steps": list(self.steps),
            "best_index": self.best_index if self.snapshots else None,
        }


_RepoLike = Union[SaddlePointRepository, Sequence[List[torch.Tensor]]]


def _select(repo: _RepoLike, select: str) -> Optional[List[torch.Tensor]]:
    if isinstance(repo, SaddlePointRepository):
        if len(repo) == 0:
            return None
        return repo.best if select == "best" else repo.snapshots[-1]
    if len(repo) == 0:
        return None
    return repo[-1]  # plain snapshot lists carry no losses: last is the only option


def aggregate_asr(
    repositories: Sequence[_RepoLike],
    device: Optional[torch.device] = None,
    select: str = "best",
) -> List[torch.Tensor]:
    """Average one snapshot per teacher into the Approximated Saddle Region.

    Parameters
    ----------
    repositories
        One :class:`SaddlePointRepository` per teacher (recommended: the lowest-loss
        snapshot of each is used, matching paper Section 3.2), or a plain list of
        snapshot-lists per teacher (no losses are available, so the last snapshot is used).
    device
        Device for the returned tensors (default: keep the first snapshot's device).
    select : {"best", "last"}
        Which snapshot to take from each repository.
    """

    if select not in {"best", "last"}:
        raise ValueError(f"select must be 'best' or 'last', got {select!r}")
    if not repositories:
        raise ValueError("Cannot aggregate an empty list of repositories.")

    chosen = [s for s in (_select(r, select) for r in repositories) if s is not None]
    if not chosen:
        raise ValueError("All repositories are empty - nothing to aggregate.")

    n = len(chosen)
    base = [t.clone().detach().float() for t in chosen[0]]
    for snap in chosen[1:]:
        for i, t in enumerate(snap):
            base[i] = base[i] + t.detach().to(base[i].device).float()

    averaged = [t / n for t in base]
    if device is not None:
        averaged = [t.to(device) for t in averaged]
    return averaged


def estimate_top_eigenvalues(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data: tuple,
    top_n: int = 4,
    use_cuda: Optional[bool] = None,
):
    """Estimate the top-``top_n`` Hessian eigenvalues using PyHessian.

    Train/eval mode and gradients of ``model`` are preserved (see
    :mod:`sprkd.hessian_utils`).
    """

    from sprkd.hessian_utils import top_eigenpairs

    return top_eigenpairs(model, criterion, data, top_n=top_n)
