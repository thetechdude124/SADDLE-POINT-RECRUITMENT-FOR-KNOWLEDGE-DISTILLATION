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


_RuleName = Literal["extreme", "magnitude", "ratio", "both"]


@dataclass
class SaddleCriterion:
    """Hyperparameters for the saddle-point detection rule.

    Parameters
    ----------
    rule : {"extreme", "magnitude", "ratio", "both"}, default ``"extreme"``
        ``"extreme"`` (default since 0.3.0): the point qualifies when ``lambda_max > 0`` and
        ``lambda_min < -tau`` with ``lambda_max`` / ``lambda_min`` computed directly
        (:func:`sprkd.hessian_utils.extreme_eigenpairs`). The other three rules act on the
        top-k-by-magnitude eigenvalues and are kept to reproduce version 0.1/0.2 behaviour.
    tau, tau_rel : float
        Threshold for ``"extreme"``: absolute ``tau`` if given, else ``tau_rel * lambda_max``.
    alpha : float
        Ratio threshold (paper's :math:`\\alpha`), used by ``"ratio"`` and ``"both"``.
    magnitude_threshold : float
        Absolute negative-eigenvalue mass threshold (paper's :math:`\\beta`).
    require_negative_eigenvalue : bool
        At least one strictly negative eigenvalue must be present.
    max_grad_norm : float or None
        If set, a step qualifies only when the gradient L2 norm is at most this value.
    """

    rule: _RuleName = "extreme"
    alpha: float = 0.4
    magnitude_threshold: float = 7.0
    require_negative_eigenvalue: bool = True
    max_grad_norm: Optional[float] = None
    tau: Optional[float] = None
    tau_rel: float = 0.05

    def tau_for(self, lambda_max: float) -> float:
        """Negative-curvature threshold: absolute ``tau`` if set, else ``tau_rel * lambda_max``."""
        return float(self.tau) if self.tau is not None else self.tau_rel * max(float(lambda_max), 0.0)


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
    eigenvalues: Optional[Sequence[float]] = None,
    criterion: Optional[SaddleCriterion] = None,
    grad_norm: Optional[float] = None,
    *,
    lambda_max: Optional[float] = None,
    lambda_min: Optional[float] = None,
) -> dict:
    """Evaluate every condition and return a dict of booleans plus the decision.

    Keys: ``magnitude``, ``ratio``, ``grad_gate`` (True when the gate passes or is off),
    ``has_negative``, ``fired`` (the final decision for ``criterion.rule``).
    """

    if criterion is None:
        criterion = SaddleCriterion()

    grad_gate = bool(criterion.max_grad_norm is None or grad_norm is None or grad_norm <= criterion.max_grad_norm)
    if criterion.rule == "extreme":
        if lambda_max is None or lambda_min is None:
            if eigenvalues is None:
                raise ValueError("rule='extreme' needs lambda_max and lambda_min (or eigenvalues to take extremes of)")
            lambda_max, lambda_min = max(float(e) for e in eigenvalues), min(float(e) for e in eigenvalues)
        tau = criterion.tau_for(lambda_max)
        out = {"lambda_max": float(lambda_max), "lambda_min": float(lambda_min), "tau": tau, "grad_gate": grad_gate,
               "has_negative": lambda_min < 0, "extreme": bool(lambda_max > 0 and lambda_min < -tau)}
        out["fired"] = bool(out["extreme"] and grad_gate)
        return out

    pos, neg, _ = _split_signs(eigenvalues or [])
    pos_mass = sum(pos)
    neg_mass = abs(sum(neg))
    out = {
        "has_negative": bool(neg),
        "ratio": bool(neg_mass >= criterion.alpha * pos_mass),
        "magnitude": bool(neg_mass >= criterion.magnitude_threshold),
        "grad_gate": grad_gate,
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
    eigenvalues: Optional[Sequence[float]] = None,
    criterion: Optional[SaddleCriterion] = None,
    grad_norm: Optional[float] = None,
    *,
    lambda_max: Optional[float] = None,
    lambda_min: Optional[float] = None,
) -> bool:
    """Return ``True`` iff the eigen-information (and optionally ``grad_norm``) qualifies."""

    return which_rules_fire(eigenvalues, criterion, grad_norm, lambda_max=lambda_max, lambda_min=lambda_min)["fired"]


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


# --------------------------------------------------------------------------- #
# Stationary-point refinement
# --------------------------------------------------------------------------- #

def refine_to_stationary(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: tuple,
    *,
    max_steps: int = 50,
    grad_tol: float = 1e-2,
    lr: float = 1e-2,
    method: str = "gn",
    solver_iters: int = 30,
    verbose: bool = False,
) -> dict:
    """Move ``model`` (in place) from a candidate snapshot toward a stationary point of the
    probe-batch loss by minimising ``0.5 * ||grad L||^2``.

    ``method="gn"`` (default): Newton/Gauss-Newton step for the root of ``grad L = 0``:
    solve ``H d = g`` with MINRES (matrix-free, indefinite ``H`` allowed), then a
    backtracking line search on the gradient norm; if no step length reduces it, fall back
    to one gradient-norm-descent step ``theta -= lr * H g``. ``method="gradnorm"``: only
    the descent step (double backward through the gradient norm). Stops when the gradient
    norm is at most ``grad_tol`` or after ``max_steps``.

    The stationary point is that of the loss on ``batch``, not of the full training loss.

    Returns a dict with ``grad_norm_before``, ``grad_norm_after``, ``loss_before``,
    ``loss_after``, ``steps``, ``n_hvp``, ``converged``, ``wall_clock_s``.
    """

    import time

    from sprkd.hessian_utils import HessianOperator, hessian_compatible

    t0 = time.time()
    rec = {"steps": 0, "n_hvp": 0, "converged": False, "method": method}
    with hessian_compatible(model, batch) as (m, b, _):
        m.eval()  # deterministic probe-batch loss (no dropout); mode restored on exit
        params = [p for p in m.parameters() if p.requires_grad]

        def flat_params():
            return torch.cat([p.detach().reshape(-1) for p in params])

        def set_params(vec):
            i = 0
            with torch.no_grad():
                for p in params:
                    n = p.numel(); p.copy_(vec[i:i + n].view_as(p)); i += n

        op = HessianOperator(m, loss_fn, b)
        rec["loss_before"] = float(op.loss.detach()); rec["grad_norm_before"] = float(op.grad_flat.norm())
        g_norm = rec["grad_norm_before"]
        for step in range(max_steps):
            if g_norm <= grad_tol:
                rec["converged"] = True
                break
            g = op.grad_flat
            theta = flat_params()
            moved = False
            if method == "gn":
                try:
                    import numpy as np
                    from scipy.sparse.linalg import LinearOperator, minres

                    dev, dt = g.device, g.dtype
                    lin = LinearOperator((op.n, op.n), matvec=lambda x: op.hvp(torch.as_tensor(np.asarray(x, dtype=np.float64).ravel(), dtype=dt, device=dev)).double().cpu().numpy(), dtype=np.float64)
                    d_np, _ = minres(lin, g.double().cpu().numpy(), maxiter=solver_iters, rtol=1e-4)
                    d = torch.as_tensor(d_np, dtype=dt, device=dev)
                    for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                        set_params(theta - alpha * d)
                        op_new = HessianOperator(m, loss_fn, b)
                        rec["n_hvp"] += op.n_hvp
                        if float(op_new.grad_flat.norm()) < g_norm:
                            op.release(); op = op_new; moved = True
                            break
                        op_new.release()
                except Exception:
                    moved = False
            if not moved:  # gradient-norm descent: grad of 0.5||g||^2 is H g, with backtracking
                set_params(theta)
                op.release(); op = HessianOperator(m, loss_fn, b)
                hg = op.hvp(op.grad_flat)
                rec["n_hvp"] += op.n_hvp
                for alpha in (lr, lr / 4, lr / 16, lr / 64, lr / 256):
                    set_params(theta - alpha * hg)
                    op_new = HessianOperator(m, loss_fn, b)
                    if float(op_new.grad_flat.norm()) < g_norm:
                        op.release(); op = op_new; moved = True
                        break
                    op_new.release()
                if not moved:  # stalled: restore and stop
                    set_params(theta)
                    op.release(); op = HessianOperator(m, loss_fn, b)
                    rec["stalled"] = True
                    rec["steps"] = step + 1
                    break
            g_norm = float(op.grad_flat.norm())
            rec["steps"] = step + 1
            if verbose:
                print(f"refine step {step + 1}: grad_norm={g_norm:.4g} loss={float(op.loss.detach()):.4f}")
        rec["converged"] = rec["converged"] or g_norm <= grad_tol
        rec["grad_norm_after"] = g_norm; rec["loss_after"] = float(op.loss.detach())
        rec["n_hvp"] += op.n_hvp
        op.release()
    rec["wall_clock_s"] = round(time.time() - t0, 3)
    return rec
