"""Saddle-point detection and Approximated Saddle Region (ASR) construction.

Two paper-faithful detection rules are exposed (paper Section 3.1, Eq. 1):

* ``"magnitude"`` (default, used in the canonical Colab notebook
  ``SPRKD_SADDLE_POINT_RECRUITMENT_FOR_KNOWLEDGE_DISTILLATION_ADITYA_DEWAN_2023``):

  .. math::

      \\Big| \\sum_{\\lambda_i < 0} \\lambda_i \\Big| \\;\\ge\\; \\beta,
      \\qquad \\beta = 7

  This is the rule used to populate ``TRUE_MALARIA_ENSEMBLE_TEACHER_SADDLE_POINTS.pth``
  and the released SPRKD checkpoints.

* ``"ratio"`` (the alpha-ratio rule from the original ISEF 2023 notebook):

  .. math::

      \\Big| \\sum_{\\lambda_i < 0} \\lambda_i \\Big| \\;\\ge\\;
      \\alpha \\, \\sum_{\\lambda_i > 0} \\lambda_i,
      \\qquad \\alpha = 0.4

* ``"both"`` (paper Equation 1 read literally): both conditions must hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Optional, Sequence

import torch


_RuleName = Literal["magnitude", "ratio", "both"]


@dataclass
class SaddleCriterion:
    """Hyperparameters for the saddle-point detection rule.

    Parameters
    ----------
    rule : {"magnitude", "ratio", "both"}, default ``"magnitude"``
        Which condition to enforce. ``"magnitude"`` matches the canonical
        Colab notebook and the released checkpoints. ``"ratio"`` matches the
        original ISEF 2023 notebook. ``"both"`` enforces paper Equation 1
        literally (the conjunction of the two).
    alpha : float
        Negative-eigenvalue magnitude *ratio* threshold (paper's
        :math:`\\alpha`, used by ``"ratio"`` and ``"both"``). Default ``0.4``.
    magnitude_threshold : float
        Lower bound on the absolute negative-eigenvalue mass (paper's
        :math:`\\beta`, used by ``"magnitude"`` and ``"both"``). Default
        ``7.0``.
    require_negative_eigenvalue : bool
        If ``True`` (default), at least one strictly negative eigenvalue
        must be present.
    """

    rule: _RuleName = "magnitude"
    alpha: float = 0.4
    magnitude_threshold: float = 7.0
    require_negative_eigenvalue: bool = True


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


def is_strong_saddle_point(
    eigenvalues: Sequence[float],
    criterion: Optional[SaddleCriterion] = None,
) -> bool:
    """Return ``True`` iff ``eigenvalues`` qualify as a strong saddle point.

    See the module docstring for the three available rules. The default rule
    matches the canonical SPRKD Colab notebook exactly:

    .. code-block:: python

        # canonical (latest notebook)
        if abs(sum(neg_eigs)) >= 7:
            ...
    """

    if criterion is None:
        criterion = SaddleCriterion()

    pos, neg, _ = _split_signs(eigenvalues)
    if criterion.require_negative_eigenvalue and not neg:
        return False

    pos_mass = sum(pos)
    neg_mass = abs(sum(neg))

    ratio_ok = neg_mass >= (criterion.alpha * pos_mass)
    magnitude_ok = neg_mass >= criterion.magnitude_threshold

    if criterion.rule == "magnitude":
        return bool(magnitude_ok)
    if criterion.rule == "ratio":
        return bool(ratio_ok)
    if criterion.rule == "both":
        return bool(ratio_ok and magnitude_ok)
    raise ValueError(f"Unknown saddle rule: {criterion.rule!r}")


@dataclass
class SaddlePointRepository:
    """A growing collection of (loss, params) snapshots, one per teacher.

    The :meth:`append` method clones-and-detaches the parameters into CPU
    storage to avoid memory pressure, matching the behaviour of the original
    SPRKD notebook implementation.
    """

    teacher_index: int
    snapshots: List[List[torch.Tensor]] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    def append(
        self,
        params: Iterable[torch.nn.Parameter],
        loss: Optional[float] = None,
    ) -> None:
        cpu_snap = [p.clone().detach().to("cpu") for p in params]
        self.snapshots.append(cpu_snap)
        self.losses.append(float(loss) if loss is not None else float("nan"))

    def __len__(self) -> int:  # noqa: D401 - magic method
        return len(self.snapshots)

    @property
    def best(self) -> List[torch.Tensor]:
        """Lowest-loss snapshot, or the most recent one if losses are unset."""

        if not self.snapshots:
            raise IndexError("SaddlePointRepository is empty.")

        finite = [(i, l) for i, l in enumerate(self.losses) if l == l]  # NaN-safe
        if not finite:
            return self.snapshots[-1]
        idx = min(finite, key=lambda kv: kv[1])[0]
        return self.snapshots[idx]


def aggregate_asr(
    repositories: Sequence[Sequence[List[torch.Tensor]]],
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Average the *last* (lowest-loss) snapshot from each teacher.

    This matches Section 3.2 of the paper: the lowest-loss saddle point per
    teacher is averaged into a single ASR.

    Parameters
    ----------
    repositories : Sequence[Sequence[List[Tensor]]]
        Either a list of lists of tensor-lists (per-teacher snapshots), or a
        dict-like mapping ``{teacher_index: List[List[Tensor]]}`` (flatten
        with ``list(d.values())`` first).
    device : torch.device, optional
        Device to materialise the resulting ASR tensors on. ``None`` keeps
        them on the device of the first teacher's snapshot.

    Returns
    -------
    List[torch.Tensor]
        One tensor per layer, averaged across teachers.
    """

    if not repositories:
        raise ValueError("Cannot aggregate an empty list of repositories.")

    last_snaps = [repo[-1] for repo in repositories if len(repo) > 0]
    if not last_snaps:
        raise ValueError("All repositories are empty - nothing to aggregate.")

    n = len(last_snaps)
    base = [t.clone().detach().float() for t in last_snaps[0]]
    for snap in last_snaps[1:]:
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

    Thin wrapper around :class:`pyhessian.hessian` that handles device
    detection so the rest of the package does not need to import PyHessian
    directly.
    """

    from pyhessian import hessian as PyHessian

    if use_cuda is None:
        use_cuda = next(model.parameters()).is_cuda

    hess = PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)
    eigenvalues, eigenvectors = hess.eigenvalues(top_n=top_n)
    return eigenvalues, eigenvectors
