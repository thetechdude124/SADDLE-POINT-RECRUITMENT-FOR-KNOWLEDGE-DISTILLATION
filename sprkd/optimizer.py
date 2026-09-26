"""The SPRKD optimizer.

A ``torch.optim.Optimizer`` subclass that implements the three SPRKD modes
described in Section 3 of the paper:

1. **Teacher mode** (``is_teacher=True``): runs an inner ``base_optimizer``
   (e.g. Adam) and, every ``saddle_steps`` iterations, evaluates the
   strong-saddle criterion on the model's Hessian. Qualifying snapshots are
   stored in :attr:`saddle_repository` together with the loss, the gradient
   norm and which rule fired.

2. **Control mode** (``is_control=True``): a thin pass-through to
   ``base_optimizer.step()`` for scratch-trained baselines.

3. **Student mode** (default): drives the model through three sub-phases:

   a. *Iterative ASR approaching* via the exponentially-decayed Euclidean
      Distance Matrix transformation (Section 3.3.1).
   b. *Negative Hessian Eigensteps (NHE)* once near the ASR (Section 3.3.2):
      a step of size ``eta`` along each negative-curvature eigenvector ``v``
      in the descent direction ``-sign(g . v) v``, reverted if the batch loss
      increases.
   c. *Gaussian Perturbed Gradient Descent (PGD)* to escape near-degenerate
      saddles (Section 3.3.2), reverted together with the NHE step if the
      pair fails to reduce the batch loss.

Every Hessian computation preserves the model's train/eval mode and its
gradients (see :mod:`sprkd.hessian_utils`). Counters for every event are
available via :meth:`SPRKD.counters`.

Default hyperparameters and their provenance (paper vs. notebook vs. this
package) are tabulated in ``neurips/04_paper_vs_code_reconciliation.md``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

import copy

from sprkd.hessian_utils import batch_loss, default_hessian_factory, extreme_eigenpairs, hessian_compatible
from sprkd.saddle import (
    SaddleCriterion,
    SaddlePointRepository,
    refine_to_stationary,
    which_rules_fire,
)


_HessianFactory = Callable[[], Any]


def _grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(total)


class SPRKD(torch.optim.Optimizer):
    """Saddle Point Recruitment for Knowledge Distillation optimizer.

    Parameters
    ----------
    params : Iterable[torch.nn.Parameter]
        Parameters to optimize.
    base_optimizer : torch.optim.Optimizer
        Inner first-order optimizer (Adam, SGD, ...). SPRKD wraps it; the
        user is responsible for constructing it on the same parameters.
    loss_fn : nn.Module
        Loss criterion - required for Hessian-vector products.
    is_teacher : bool, default False
        Enable teacher-mode saddle tracking.
    is_control : bool, default False
        Pass-through mode (no SPRKD logic).
    teacher_saddle_points : list of torch.Tensor, optional
        ASR tensors (one per ``params`` element). Required in student mode.
    saddle_steps : int or None, default 50
        Stride between saddle checks (teacher mode only). ``None`` disables
        saddle checks (student mode).
    saddle_step_limit : int or None, default None
        If supplied, stop tracking saddle points after this many global
        steps (teacher mode).
    saddle_criterion : SaddleCriterion, optional
        Detection thresholds. Defaults to ``SaddleCriterion()`` (``"extreme"``
        rule: ``lambda_max > 0`` and ``lambda_min < -0.05 lambda_max``, no
        gradient gate). Use ``rule="magnitude"`` to reproduce version 0.1/0.2.
    saddle_top_k : int or None, default None
        Keep only the ``top_k`` lowest-loss snapshots in the repository.
    epsilon : float, default 1e-3
        Maximum allowed Euclidean distance between student and ASR before
        the iterative-approach phase terminates (the paper text says 0.1; the
        notebook and released checkpoints used 1e-3).
    pgd_grad_threshold : float, default 0.02
        Gradient L2-norm threshold below which a parameter tensor is flagged
        as stagnating (paper's ``j``; the notebook's final run used 0.02).
    pgd_delta : float, default 0.25
        Mean ``| ||p|| - ||T|| |`` gap to the ASR above which perturbations
        are allowed (notebook's final run: 0.25; the previous package default
        of 5.0 was effectively unreachable).
    pgd_epoch_limit : int, default 100
        Disable perturbations after this many epochs.
    pgd_perturb_variance : float, default 0.1
        Variance of the Gaussian perturbation ``xi ~ N(0, 0.1)`` (paper Sec.
        3.3.2). Set to 0 to disable PGD.
    max_nhe_steps : int, default 50
        Cap on NHE attempts per training run. Set to 0 to disable NHE.
    cooldown_steps : int, default 20
        Minimum gap between successive perturbation events.
    nhe_step_mode : {"adaptive", "fixed"}, default ``"adaptive"``
        ``"fixed"``: step length ``eta = nhe_step_size`` along each negative
        eigenvector. ``"adaptive"``: ``eta = min(nhe_step_size, |lambda|)``,
        the Nesterov-Polyak choice ``|lambda| / rho`` with the Hessian
        Lipschitz constant ``rho`` taken as 1 and capped by ``nhe_step_size``.
    nhe_step_size : float, default 0.1
        Step length (fixed mode) or cap (adaptive mode).
    n_top_eigs : int, default 4
        Number of leading eigenvalues to compute when checking saddles.
    n_nhe_eigs : int, default 2
        Number of leading eigenpairs to compute for NHE when
        ``nhe_direction="topk"``; every negative one is used.
    nhe_direction : {"lambda_min", "topk"}, default ``"lambda_min"``
        ``"lambda_min"`` computes the most negative eigenpair directly and steps
        along it whenever ``lambda_min < -tau``; ``"topk"`` reproduces the 0.2
        behaviour (negative eigenvalues among the top-k by magnitude).
    saddle_refine : bool, default False
        Teacher mode: refine each candidate to a stationary point of the probe
        batch loss (:func:`sprkd.saddle.refine_to_stationary`) on a copy of the
        model, re-run the eigen check, and record it only if
        ``grad_norm <= refine_grad_tol`` and ``lambda_min < -tau``.
    refine_grad_tol_mode : {"fixed", "quantile"}
        ``"fixed"`` uses ``refine_grad_tol`` (what the paper's definition of a
        saddle point, grad = 0, requires); ``"quantile"`` uses the
        ``refine_grad_tol_quantile`` quantile of the gradient norms seen so far
        on the trajectory (a relative notion, for exploration only).
    revert_on_increase : bool, default True
        Undo an NHE + PGD event if the batch loss did not decrease.
    hessian_factory : callable, optional
        Override the PyHessian builder; primarily useful for testing.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        *,
        is_teacher: bool = False,
        is_control: bool = False,
        teacher_saddle_points: Optional[Sequence[torch.Tensor]] = None,
        saddle_steps: Optional[int] = 50,
        saddle_step_limit: Optional[int] = None,
        saddle_criterion: Optional[SaddleCriterion] = None,
        saddle_top_k: Optional[int] = None,
        epsilon: float = 1e-3,
        pgd_grad_threshold: float = 0.02,
        pgd_delta: float = 0.25,
        pgd_epoch_limit: int = 100,
        pgd_perturb_variance: float = 0.1,
        max_nhe_steps: int = 50,
        cooldown_steps: int = 20,
        nhe_step_mode: str = "adaptive",
        nhe_step_size: float = 0.1,
        n_top_eigs: int = 4,
        n_nhe_eigs: int = 2,
        nhe_direction: str = "lambda_min",
        revert_on_increase: bool = True,
        saddle_refine: bool = False,
        refine_max_steps: int = 30,
        refine_grad_tol: float = 1e-2,
        refine_grad_tol_mode: str = "fixed",
        refine_grad_tol_quantile: float = 0.1,
        refine_lr: float = 1e-2,
        refine_loss_tol: float = 0.05,
        hessian_factory: Optional[_HessianFactory] = None,
    ):
        if is_teacher and is_control:
            raise ValueError("is_teacher and is_control are mutually exclusive.")
        if not is_teacher and not is_control and teacher_saddle_points is None:
            raise ValueError(
                "teacher_saddle_points must be provided when SPRKD is used "
                "in student mode."
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if saddle_steps is not None and saddle_steps <= 0:
            raise ValueError(f"saddle_steps must be > 0 or None, got {saddle_steps}")
        if saddle_step_limit is not None and saddle_step_limit <= 0:
            raise ValueError(
                f"saddle_step_limit must be > 0 or None, got {saddle_step_limit}"
            )
        if nhe_step_mode not in {"adaptive", "fixed"}:
            raise ValueError(
                f"nhe_step_mode must be 'adaptive' or 'fixed', got {nhe_step_mode!r}"
            )
        if pgd_perturb_variance < 0:
            raise ValueError(
                f"pgd_perturb_variance must be >= 0, got {pgd_perturb_variance}"
            )
        if nhe_step_size <= 0:
            raise ValueError(f"nhe_step_size must be > 0, got {nhe_step_size}")
        if nhe_direction not in {"lambda_min", "topk"}:
            raise ValueError(f"nhe_direction must be 'lambda_min' or 'topk', got {nhe_direction!r}")
        if refine_grad_tol_mode not in {"fixed", "quantile"}:
            raise ValueError(f"refine_grad_tol_mode must be 'fixed' or 'quantile', got {refine_grad_tol_mode!r}")

        defaults: dict = dict(
            is_teacher=is_teacher,
            is_control=is_control,
            saddle_steps=saddle_steps,
            saddle_step_limit=saddle_step_limit,
            epsilon=epsilon,
            pgd_grad_threshold=pgd_grad_threshold,
            pgd_delta=pgd_delta,
            pgd_epoch_limit=pgd_epoch_limit,
            pgd_perturb_variance=pgd_perturb_variance,
            max_nhe_steps=max_nhe_steps,
            cooldown_steps=cooldown_steps,
            nhe_step_mode=nhe_step_mode,
            nhe_step_size=nhe_step_size,
            n_top_eigs=n_top_eigs,
            n_nhe_eigs=n_nhe_eigs,
            nhe_direction=nhe_direction,
            revert_on_increase=revert_on_increase,
            saddle_refine=saddle_refine,
            refine_max_steps=refine_max_steps,
            refine_grad_tol=refine_grad_tol,
            refine_grad_tol_mode=refine_grad_tol_mode,
            refine_grad_tol_quantile=refine_grad_tol_quantile,
            refine_lr=refine_lr,
            refine_loss_tol=refine_loss_tol,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn
        self.saddle_criterion = saddle_criterion or SaddleCriterion()
        self._hessian_factory = hessian_factory or default_hessian_factory

        self.teacher_saddle_points: List[torch.Tensor] = (
            list(teacher_saddle_points) if teacher_saddle_points is not None else []
        )

        # Persistent state (per-optimizer, not per-parameter).
        self.saddle_repository = SaddlePointRepository(teacher_index=0, top_k=saddle_top_k)
        self._step_count: int = 0
        self._allow_targeting: dict[int, bool] = {}
        self._cooldown: int = cooldown_steps
        self._stored_loss: float = 0.0
        self._tm_finished_step: Optional[int] = None
        # Event counters.
        self._n_nhe_taken: int = 0          # NHE attempts (Hessian computed)
        self._n_nhe_applied: int = 0        # attempts that changed parameters
        self._n_nhe_no_negative: int = 0    # attempts with no negative eigenvalue
        self._n_nhe_reverted: int = 0       # NHE-only reverts (loss increased)
        self._n_pgd_fired: int = 0          # Gaussian perturbations applied and kept
        self._n_pgd_reverted: int = 0       # NHE + PGD events undone
        self._n_pgd_considered: int = 0     # trigger conditions met
        self._nhe_eigenvalues: List[float] = []
        self.events: List[dict] = []        # one record per NHE / PGD event (for inspection)
        self._trajectory_grad_norms: List[float] = []   # for the quantile grad_tol mode
        self._n_candidates: int = 0         # checks whose eigen rule fired before refinement
        self._n_refined: int = 0            # candidates that went through refinement
        self._n_refine_rejected: int = 0    # refined candidates that failed the verification

    # ------------------------------------------------------------------ utils
    @property
    def step_count(self) -> int:
        return self._step_count

    def at_asr(self) -> bool:
        """Return ``True`` iff every parameter has reached the ASR within ``epsilon``."""

        if not self._allow_targeting:
            return False
        return not any(self._allow_targeting.values())

    def counters(self) -> dict:
        """Event counters for logging and ablations."""

        return {
            "step_count": self._step_count,
            "tm_finished_step": self._tm_finished_step,
            "nhe_taken": self._n_nhe_taken,
            "nhe_applied": self._n_nhe_applied,
            "nhe_no_negative": self._n_nhe_no_negative,
            "nhe_reverted": self._n_nhe_reverted,
            "pgd_considered": self._n_pgd_considered,
            "pgd_fired": self._n_pgd_fired,
            "pgd_reverted": self._n_pgd_reverted,
            "saddles_checked": self.saddle_repository.n_checked,
            "saddle_candidates": self._n_candidates,
            "saddles_refined": self._n_refined,
            "saddles_refine_rejected": self._n_refine_rejected,
            "saddles_recorded": len(self.saddle_repository),
        }

    def _all_params(self):
        for group in self.param_groups:
            for p in group["params"]:
                yield p, group

    def _zero_grad_inner(self):
        self.base_optimizer.zero_grad(set_to_none=False)

    # -------------------------------------------------------------- main step
    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
        *,
        model: Optional[nn.Module] = None,
        current_loss: Optional[torch.Tensor] = None,
        data_batch: Optional[tuple] = None,
    ) -> Optional[torch.Tensor]:
        """Perform a single SPRKD step.

        The signature deliberately mirrors ``torch.optim.Optimizer.step``: a
        plain ``optimizer.step()`` (or ``step(closure)``) call routes through
        ``base_optimizer`` only, replicating control-mode behaviour. The
        SPRKD-specific logic activates when ``model`` and ``current_loss``
        are supplied (training loops in :mod:`sprkd.training` do this for
        you).
        """

        loss = closure() if closure is not None else None
        self._step_count += 1

        # Fast path: no model context -> behave exactly like base optimizer.
        if model is None or current_loss is None:
            self.base_optimizer.step()
            return loss

        for group in self.param_groups:
            if group["is_control"]:
                self.base_optimizer.step()
                continue

            if group["is_teacher"]:
                grad_norm = _grad_norm(group["params"])
                self._trajectory_grad_norms.append(grad_norm)
                self.base_optimizer.step()
                limit = group["saddle_step_limit"]
                if (
                    group["saddle_steps"] is not None
                    and self._step_count % group["saddle_steps"] == 0
                    and (limit is None or self._step_count < limit)
                ):
                    self._maybe_record_saddle(
                        group=group,
                        model=model,
                        data_batch=data_batch,
                        loss_value=float(current_loss.detach().cpu()),
                        grad_norm=grad_norm,
                    )
                continue

            # Student mode.
            if not self._allow_targeting:
                self._allow_targeting = {i: True for i, _ in enumerate(group["params"])}

            avg_distance = self._student_average_distance(group)

            if any(self._allow_targeting.values()):
                self._apply_transformation_matrix(group)
                if not any(self._allow_targeting.values()):
                    self._tm_finished_step = self._step_count
            else:
                if self._tm_finished_step is None:
                    self._tm_finished_step = self._step_count
                self.base_optimizer.step()
                self._maybe_apply_perturbation(
                    group=group,
                    model=model,
                    data_batch=data_batch,
                    avg_distance=avg_distance,
                    current_loss=float(current_loss.detach().cpu()),
                )

            if self._cooldown > 0:
                self._cooldown -= 1

        return loss

    # ----------------------------------------------------- teacher-mode logic
    def _hessian_compat_model(self, model: nn.Module, data_batch: Optional[tuple]):
        """Backward-compatible alias for :func:`sprkd.hessian_utils.hessian_compatible`."""

        return hessian_compatible(model, data_batch)

    def _maybe_record_saddle(
        self,
        *,
        group: dict,
        model: nn.Module,
        data_batch: Optional[tuple],
        loss_value: float,
        grad_norm: Optional[float] = None,
    ) -> None:
        if data_batch is None:
            try:
                data_batch = next(iter(model.dls.train))
            except AttributeError as e:
                raise RuntimeError(
                    "Teacher-mode SPRKD requires a `data_batch` argument or a "
                    "model with `.dls.train` (fastai-style)."
                ) from e

        self.saddle_repository.n_checked += 1
        outcome = self._eigen_check(group, model, data_batch, grad_norm)
        outcome["rule"] = self.saddle_criterion.rule
        if not outcome["fired"]:
            return
        self._n_candidates += 1
        params_to_store = group["params"]
        if group["saddle_refine"]:
            outcome = self._refine_candidate(group, model, data_batch, outcome, loss_value)
            if not outcome["verified"]:
                self._n_refine_rejected += 1
                return
            params_to_store = outcome.pop("_refined_params")
            loss_value = outcome["loss_after"]
            grad_norm = outcome["grad_norm_after"]
        self.saddle_repository.append(
            params_to_store,
            loss=loss_value,
            grad_norm=grad_norm,
            step=self._step_count,
            rule=outcome,
        )

    def _eigen_check(self, group: dict, model: nn.Module, data_batch: tuple, grad_norm: Optional[float]) -> dict:
        """Evaluate the saddle criterion at the model's current parameters."""

        if self.saddle_criterion.rule == "extreme":
            ext = extreme_eigenpairs(model, self.loss_fn, data_batch)
            outcome = which_rules_fire(None, self.saddle_criterion, grad_norm,
                                       lambda_max=ext["lambda_max"], lambda_min=ext["lambda_min"])
            outcome["n_hvp"] = ext["n_hvp"]
            outcome["eigen_method"] = ext["method"]
            return outcome
        with hessian_compatible(model, data_batch) as (m, batch, use_cuda):
            hess = self._hessian_factory(m, self.loss_fn, batch, use_cuda)
            eigenvalues, _ = hess.eigenvalues(top_n=group["n_top_eigs"])
        outcome = which_rules_fire(eigenvalues, self.saddle_criterion, grad_norm)
        outcome["eigenvalues"] = [float(e) for e in eigenvalues]
        return outcome

    def _current_grad_tol(self, group: dict) -> float:
        if group["refine_grad_tol_mode"] == "quantile" and len(self._trajectory_grad_norms) >= 5:
            return float(torch.tensor(self._trajectory_grad_norms).quantile(group["refine_grad_tol_quantile"]))
        return float(group["refine_grad_tol"])

    def _refine_candidate(self, group: dict, model: nn.Module, data_batch: tuple, candidate: dict, loss_value: float) -> dict:
        """Refine a candidate to a stationary point on a copy of the model and re-verify it.

        A verified saddle has ``grad_norm <= grad_tol`` and ``lambda_min < -tau`` after
        refinement. The record keeps loss and gradient norm before/after, both extreme
        eigenvalues, the number of refinement steps and HVPs, wall-clock, and whether the
        loss drifted by more than ``refine_loss_tol``.
        """

        self._n_refined += 1
        grad_tol = self._current_grad_tol(group)
        work = copy.deepcopy(model)
        rec = refine_to_stationary(work, self.loss_fn, data_batch, max_steps=group["refine_max_steps"],
                                   grad_tol=grad_tol, lr=group["refine_lr"])
        ext = extreme_eigenpairs(work, self.loss_fn, data_batch)
        tau = self.saddle_criterion.tau_for(ext["lambda_max"])
        verified = bool(rec["grad_norm_after"] <= grad_tol and ext["lambda_min"] < -tau)
        out = dict(candidate)
        out.update({
            "refined": True, "grad_tol": grad_tol, "grad_tol_mode": group["refine_grad_tol_mode"],
            "candidate_loss": loss_value, "loss_before": rec["loss_before"], "loss_after": rec["loss_after"],
            "loss_drift": rec["loss_after"] - rec["loss_before"],
            "loss_within_tol": abs(rec["loss_after"] - rec["loss_before"]) <= group["refine_loss_tol"],
            "grad_norm_before": rec["grad_norm_before"], "grad_norm_after": rec["grad_norm_after"],
            "lambda_max_before": candidate.get("lambda_max"), "lambda_min_before": candidate.get("lambda_min"),
            "lambda_max_after": ext["lambda_max"], "lambda_min_after": ext["lambda_min"], "tau_after": tau,
            "refine_steps": rec["steps"], "refine_n_hvp": rec["n_hvp"] + ext["n_hvp"], "refine_converged": rec["converged"],
            "refine_wall_clock_s": rec["wall_clock_s"], "verified": verified,
            "_refined_params": [p.detach().clone() for p in work.parameters()],
        })
        del work
        return out

    # ----------------------------------------------------- student-mode logic
    def _student_average_distance(self, group: dict) -> torch.Tensor:
        total = torch.zeros(())
        if not self.teacher_saddle_points:
            return total
        params = group["params"]
        for p, sp in zip(params, self.teacher_saddle_points):
            total = total + torch.abs(
                torch.linalg.norm(p.detach()) - torch.linalg.norm(sp.detach())
            ).to(total.device)
        return total / len(params)

    @torch.no_grad()
    def _apply_transformation_matrix(self, group: dict) -> None:
        params = group["params"]
        eps = group["epsilon"]
        for i, (p, sp) in enumerate(zip(params, self.teacher_saddle_points)):
            sp = sp.to(p.device, dtype=p.dtype)
            distance = self._diag_euclidean_distance(p, sp)
            if not torch.any(distance > eps).item() or not self._allow_targeting.get(i, True):
                self._allow_targeting[i] = False
                continue
            tm = torch.div(sp, p.where(p != 0, torch.tensor(1e-8, device=p.device)))
            # Decay as implemented in the notebook that produced the released
            # checkpoints: l = 1 - 2^(-t/10) / 2 (the paper text omits the /2).
            weight = -2.0 ** (-self._step_count / 10.0) / 2.0 + 1.0
            p.data = p.data.mul(weight * tm)

    @staticmethod
    def _diag_euclidean_distance(p: torch.Tensor, sp: torch.Tensor) -> torch.Tensor:
        if p.dim() < 2:
            return torch.diagonal(torch.cdist(p.unsqueeze(1), sp.unsqueeze(1)))
        if p.dim() == 2:
            return torch.diagonal(torch.cdist(p, sp))
        return torch.diagonal(torch.diagonal(torch.cdist(p, sp)))

    def _maybe_apply_perturbation(
        self,
        *,
        group: dict,
        model: nn.Module,
        data_batch: Optional[tuple],
        avg_distance: torch.Tensor,
        current_loss: float,
    ) -> None:
        """NHE + Gaussian PGD event with verification and revert (paper Sec. 3.3.2).

        Trigger (per the canonical notebook): some parameter tensor has
        ``||grad|| < pgd_grad_threshold``, the mean norm gap to the ASR exceeds
        ``pgd_delta``, the epoch is below ``pgd_epoch_limit``, the cooldown has
        elapsed, the ASR has been reached, and the loss has decreased by at
        least 0.002 since the previous event. Then:

        1. snapshot all parameters;
        2. NHE step (if budget remains): ``theta -= eta * sign(g . v) * v`` for
           every negative eigenpair, reverted on its own if the batch loss rose;
        3. Gaussian perturbation of the flagged tensor, ``sqrt(variance) * N(0, 1)``;
        4. if the batch loss after (2)+(3) is not below the pre-event loss, revert
           everything to the snapshot.
        """

        if self._cooldown > 0:
            return
        if any(self._allow_targeting.values()):
            return

        steps_per_epoch = max(1, getattr(model, "_steps_per_epoch", 1))
        if self._step_count / steps_per_epoch >= group["pgd_epoch_limit"]:
            return
        if data_batch is None:
            try:
                data_batch = next(iter(model.dls.train))
            except AttributeError:
                return

        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            grad_norm = torch.linalg.norm(p.grad.detach())
            if grad_norm.item() >= group["pgd_grad_threshold"]:
                continue
            if avg_distance.item() <= group["pgd_delta"]:
                continue
            if self._stored_loss != 0.0 and self._stored_loss - current_loss < 0.002:
                continue

            self._n_pgd_considered += 1
            snapshot = [q.detach().clone() for q in group["params"]]
            pre_loss = batch_loss(model, self.loss_fn, data_batch)

            self._negative_hessian_eigenstep(group=group, model=model, data_batch=data_batch)

            std = math.sqrt(abs(group["pgd_perturb_variance"]))
            if std > 0:
                with torch.no_grad():
                    p.data = p.data + std * torch.randn_like(p.data)

            post_loss = batch_loss(model, self.loss_fn, data_batch)
            reverted = bool(group["revert_on_increase"] and post_loss >= pre_loss)
            self.events.append({
                "kind": "pgd", "step": self._step_count, "param_index": i, "grad_norm": float(grad_norm),
                "avg_distance": float(avg_distance), "std": std, "pre_loss": pre_loss, "post_loss": post_loss,
                "reverted": reverted,
            })
            if reverted:
                with torch.no_grad():
                    for q, s in zip(group["params"], snapshot):
                        q.data.copy_(s)
                self._n_pgd_reverted += 1
            else:
                self._n_pgd_fired += 1

            self._stored_loss = current_loss
            self._cooldown = group["cooldown_steps"]
            return

    def _negative_hessian_eigenstep(
        self,
        *,
        group: dict,
        model: nn.Module,
        data_batch: Optional[tuple],
    ) -> bool:
        """Negative-curvature descent step along every negative top eigenvector.

        For each eigenpair ``(lambda, v)`` with ``lambda < 0`` (``v`` a unit
        vector over all parameters), move ``theta <- theta - eta * sign(g . v) * v``
        so that the first-order term decreases; second-order descent follows from
        ``lambda < 0`` (Nesterov & Polyak 2006; Carmon et al. 2018). The step is
        reverted if the batch loss increases. Returns ``True`` if parameters were
        changed and kept.
        """

        if self._n_nhe_taken >= group["max_nhe_steps"]:
            return False

        if data_batch is None:
            try:
                data_batch = next(iter(model.dls.train))
            except AttributeError:
                return False

        self._n_nhe_taken += 1
        if group["nhe_direction"] == "lambda_min":
            ext = extreme_eigenpairs(model, self.loss_fn, data_batch)
            tau = self.saddle_criterion.tau_for(ext["lambda_max"])
            top_eigs = [ext["lambda_max"], ext["lambda_min"]]
            negatives = [(ext["lambda_min"], ext["v_min"])] if ext["lambda_min"] < -tau else []
        else:
            with hessian_compatible(model, data_batch) as (m, batch, use_cuda):
                hess = self._hessian_factory(m, self.loss_fn, batch, use_cuda)
                top_eigs, top_vecs = hess.eigenvalues(top_n=group["n_nhe_eigs"])
            negatives = [(float(ev), vec) for ev, vec in zip(top_eigs, top_vecs) if float(ev) < 0]
        if not negatives:
            self._n_nhe_no_negative += 1
            self.events.append({"kind": "nhe", "step": self._step_count, "eigenvalues": [float(e) for e in top_eigs],
                                "n_negative": 0, "applied": False, "reverted": False})
            return False

        params = group["params"]
        snapshot = [q.detach().clone() for q in params]
        pre_loss = batch_loss(model, self.loss_fn, data_batch)

        with torch.no_grad():
            for lam, v_layers in negatives:
                vecs = [
                    (v if isinstance(v, torch.Tensor) else torch.as_tensor(v)).to(p.device, dtype=p.dtype)
                    for p, v in zip(params, v_layers)
                ]
                norm = math.sqrt(sum(float(v.float().pow(2).sum()) for v in vecs)) or 1.0
                gv = sum(
                    float((p.grad.detach().float() * v.float()).sum())
                    for p, v in zip(params, vecs)
                    if p.grad is not None
                )
                sign = 1.0 if gv >= 0 else -1.0
                if group["nhe_step_mode"] == "adaptive":
                    eta = min(float(group["nhe_step_size"]), abs(lam))
                else:
                    eta = float(group["nhe_step_size"])
                for p, v in zip(params, vecs):
                    p.data.add_(v, alpha=-eta * sign / norm)
                self._nhe_eigenvalues.append(lam)

        post_loss = batch_loss(model, self.loss_fn, data_batch)
        reverted = bool(group["revert_on_increase"] and post_loss > pre_loss)
        self.events.append({
            "kind": "nhe", "step": self._step_count, "eigenvalues": [float(e) for e in top_eigs],
            "n_negative": len(negatives), "pre_loss": pre_loss, "post_loss": post_loss, "reverted": reverted,
        })
        if reverted:
            with torch.no_grad():
                for q, s in zip(params, snapshot):
                    q.data.copy_(s)
            self._n_nhe_reverted += 1
            return False
        self._n_nhe_applied += 1
        return True

    # ----------------------------------------------------- standard plumbing
    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        super().zero_grad(set_to_none=set_to_none)
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:  # type: ignore[override]
        sd = super().state_dict()
        sd["sprkd_extra"] = {
            "step_count": self._step_count,
            "allow_targeting": dict(self._allow_targeting),
            "cooldown": self._cooldown,
            "n_nhe_taken": self._n_nhe_taken,
            "stored_loss": self._stored_loss,
            "n_saddles_recorded": len(self.saddle_repository),
            "counters": self.counters(),
        }
        return sd
