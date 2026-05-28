"""The SPRKD optimizer.

A ``torch.optim.Optimizer`` subclass that implements the three SPRKD modes
described in Section 3 of the paper:

1. **Teacher mode** (``is_teacher=True``): runs an inner ``base_optimizer``
   (e.g. Adam) and, every ``saddle_steps`` iterations, evaluates the
   strong-saddle criterion on the model's Hessian. Qualifying snapshots are
   stored in :attr:`saddle_repository`.

2. **Control mode** (``is_control=True``): a thin pass-through to
   ``base_optimizer.step()`` for scratch-trained baselines.

3. **Student mode** (default): drives the model through three sub-phases:

   a. *Iterative ASR approaching* via the exponentially-decayed Euclidean
      Distance Matrix transformation (Section 3.3.1).
   b. *Negative Hessian Eigensteps (NHE)* once near the ASR (Section 3.3.2).
   c. *Gaussian Perturbed Gradient Descent (PGD)* to escape near-degenerate
      saddles (Section 3.3.2).

The mathematical conventions follow the paper exactly; deviations are noted
inline.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Callable, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from sprkd.saddle import (
    SaddleCriterion,
    SaddlePointRepository,
    is_strong_saddle_point,
)


_HessianFactory = Callable[[], Any]


def _default_hessian_factory(
    model: nn.Module, criterion: nn.Module, data: tuple, use_cuda: bool
):
    """Build a PyHessian object, transparently handling MPS / CPU / CUDA.

    PyHessian (upstream) only knows ``cuda`` and ``cpu``. On Apple Silicon
    the caller must put the model on CPU before calling this factory; see
    :meth:`SPRKD._with_hessian_compatible_model`.
    """

    from pyhessian import hessian as PyHessian

    return PyHessian(model=model, criterion=criterion, data=data, cuda=use_cuda)


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
        If supplied, *stop* tracking saddle points after this many global
        steps (teacher mode). Matches the ``saddle_step_limit`` argument
        of the canonical Colab notebook ``EXPERIMENTAL_MODEL_EVALUATIONS``.
    saddle_criterion : SaddleCriterion, optional
        Detection thresholds. Defaults to ``SaddleCriterion()`` which uses
        the ``"magnitude"`` rule (``|sum(neg)| >= 7``) - the rule used by
        the canonical Colab notebook and the released checkpoints.
    epsilon : float, default 1e-3
        Maximum allowed Euclidean distance between student and ASR before
        the iterative-approach phase terminates (matches the
        ``epsilon = 10e-3`` default in the canonical Colab).
    pgd_grad_threshold : float, default 0.01
        Gradient L2-norm threshold below which the student is flagged as
        stagnating (paper's ``j``; canonical Colab default).
    pgd_delta : float, default 5.0
        Average ASR distance above which PGD perturbations are still allowed.
    pgd_epoch_limit : int, default 100
        Disable PGD perturbations after this many epochs.
    pgd_perturb_variance : float, default 0.1
        Variance of the Gaussian perturbation; the paper specifies
        ``xi ~ N(0, 0.1)`` (Section 3.3.2). The perturbation magnitude is
        ``sqrt(pgd_perturb_variance) * randn_like(param)`` to match the
        canonical Colab ``param + sqrt(0.1) * randn_like(param)``.
    max_nhe_steps : int, default 50
        Cap on Negative Hessian Eigensteps per training run.
    cooldown_steps : int, default 20
        Minimum gap between successive perturbations.
    nhe_step_mode : {"adaptive", "fixed"}, default ``"adaptive"``
        - ``"adaptive"``: step size is ``1 / |lambda_neg|`` (canonical Colab).
        - ``"fixed"``: step size is ``nhe_step_size`` (legacy notebook).
    nhe_step_size : float, default 0.1
        Used only when ``nhe_step_mode == "fixed"``.
    n_top_eigs : int, default 4
        Number of leading eigenvalues to compute when checking saddles.
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
        epsilon: float = 1e-3,
        pgd_grad_threshold: float = 0.01,
        pgd_delta: float = 5.0,
        pgd_epoch_limit: int = 100,
        pgd_perturb_variance: float = 0.1,
        max_nhe_steps: int = 50,
        cooldown_steps: int = 20,
        nhe_step_mode: str = "adaptive",
        nhe_step_size: float = 0.1,
        n_top_eigs: int = 4,
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
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn
        self.saddle_criterion = saddle_criterion or SaddleCriterion()
        self._hessian_factory = hessian_factory or _default_hessian_factory

        self.teacher_saddle_points: List[torch.Tensor] = (
            list(teacher_saddle_points) if teacher_saddle_points is not None else []
        )

        # Persistent state (per-optimizer, not per-parameter).
        self.saddle_repository = SaddlePointRepository(teacher_index=0)
        self._step_count: int = 0
        self._allow_targeting: dict[int, bool] = {}
        self._cooldown: int = cooldown_steps
        self._n_nhe_taken: int = 0
        self._param_history_pgd: dict[int, torch.Tensor] = {}
        self._stored_loss: float = 0.0

    # ------------------------------------------------------------------ utils
    @property
    def step_count(self) -> int:
        return self._step_count

    def at_asr(self) -> bool:
        """Return ``True`` iff every parameter has reached the ASR within ``epsilon``."""

        if not self._allow_targeting:
            return False
        return not any(self._allow_targeting.values())

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
                    )
                continue

            # Student mode.
            if not self._allow_targeting:
                self._allow_targeting = {i: True for i, _ in enumerate(group["params"])}

            avg_distance = self._student_average_distance(group)

            if any(self._allow_targeting.values()):
                self._apply_transformation_matrix(group)
            else:
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
    @contextmanager
    def _hessian_compat_model(self, model: nn.Module, data_batch: Optional[tuple]):
        """Yield a (model, data_batch, use_cuda) triple safe for PyHessian.

        PyHessian only supports cuda or cpu; on MPS we transparently move
        model + batch to CPU for the duration of the call and restore the
        original device afterwards.
        """

        original = next(model.parameters()).device
        moved = False
        try:
            if original.type == "mps":
                model.to("cpu")
                moved = True
                if data_batch is not None and isinstance(data_batch, (tuple, list)):
                    data_batch = tuple(
                        d.to("cpu") if hasattr(d, "to") else d for d in data_batch
                    )
                use_cuda = False
            else:
                use_cuda = original.type == "cuda"
            yield model, data_batch, use_cuda
        finally:
            if moved:
                model.to(original)

    def _maybe_record_saddle(
        self,
        *,
        group: dict,
        model: nn.Module,
        data_batch: Optional[tuple],
        loss_value: float,
    ) -> None:
        if data_batch is None:
            try:
                data_batch = next(iter(model.dls.train))
            except AttributeError as e:
                raise RuntimeError(
                    "Teacher-mode SPRKD requires a `data_batch` argument or a "
                    "model with `.dls.train` (fastai-style)."
                ) from e

        with self._hessian_compat_model(model, data_batch) as (m, batch, use_cuda):
            hess = self._hessian_factory(m, self.loss_fn, batch, use_cuda)
            eigenvalues, _ = hess.eigenvalues(top_n=group["n_top_eigs"])

        if is_strong_saddle_point(eigenvalues, criterion=self.saddle_criterion):
            self.saddle_repository.append(group["params"], loss=loss_value)

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
        """NHE + Gaussian PGD perturbation per the canonical Colab notebook.

        Mirrors ``perturbedGD()`` from
        ``SPRKD_SADDLE_POINT_RECRUITMENT_FOR_KNOWLEDGE_DISTILLATION_ADITYA_DEWAN_2023.ipynb``:

        1. trigger only if ``|grad| < grad_threshold``,
           ``avg_distance > pgd_delta``, ``epoch < pgd_epoch_limit``,
           cooldown elapsed, and ASR has been reached;
        2. NHE step (if budget remains) using
           ``weight = 1 / |lambda_neg|`` (or fixed in legacy mode);
        3. Gaussian perturbation with variance 0.1 (paper Sec. 3.3.2).
        """

        if self._cooldown > 0:
            return
        if any(self._allow_targeting.values()):
            return

        steps_per_epoch = max(1, getattr(model, "_steps_per_epoch", 1))
        if self._step_count / steps_per_epoch >= group["pgd_epoch_limit"]:
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

            self._param_history_pgd[i] = p.detach().clone()

            self._negative_hessian_eigenstep(group=group, model=model, data_batch=data_batch)

            std = math.sqrt(abs(group["pgd_perturb_variance"]))
            with torch.no_grad():
                p.data = p.data + std * torch.randn_like(p.data)

            self._stored_loss = current_loss
            self._cooldown = group["cooldown_steps"]
            return

    def _negative_hessian_eigenstep(
        self,
        *,
        group: dict,
        model: nn.Module,
        data_batch: Optional[tuple],
    ) -> None:
        """Take a single NHE step along the largest-magnitude negative direction.

        Mirrors ``negativeHessianEigensteps()`` in the canonical notebook:

        .. code-block:: python

            weight = 1 / largest_negative_eigenvalue           # adaptive
            param.data -= weight * grad * v * v                # broadcast

        and is bounded by ``max_nhe_steps`` to match the upper bound used
        by the released SPRKD checkpoints.
        """

        if self._n_nhe_taken >= group["max_nhe_steps"]:
            return

        if data_batch is None:
            try:
                data_batch = next(iter(model.dls.train))
            except AttributeError:
                return

        with self._hessian_compat_model(model, data_batch) as (m, batch, use_cuda):
            hess = self._hessian_factory(m, self.loss_fn, batch, use_cuda)
            top_eigs, top_vecs = hess.eigenvalues(top_n=2)

        if not top_eigs:
            return
        ev_index = top_eigs.index(min(top_eigs))
        lambda_neg = float(top_eigs[ev_index])
        if lambda_neg >= 0:
            return  # only negative-curvature directions

        v_layers = top_vecs[ev_index]
        if group["nhe_step_mode"] == "adaptive":
            weight = 1.0 / abs(lambda_neg)
        else:
            weight = float(group["nhe_step_size"])

        with torch.no_grad():
            for p, v in zip(group["params"], v_layers):
                if p.grad is None:
                    continue
                v_t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                v_t = v_t.to(p.device, dtype=p.dtype)
                # paper formula: theta <- theta - weight * grad * (v * v)
                # element-wise; canonical Colab uses the same.
                grad = p.grad.detach()
                step = grad.mul(v_t).mul(v_t)
                p.data = p.data - weight * step

        self._n_nhe_taken += 1

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
        }
        return sd
