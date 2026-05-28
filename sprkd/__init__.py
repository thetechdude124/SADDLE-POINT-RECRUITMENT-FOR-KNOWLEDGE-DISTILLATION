"""SPRKD: Saddle Point Recruitment for Knowledge Distillation.

A second-order, landscape-aware knowledge distillation algorithm that
re-frames distillation from teacher-output replication to saddle-region
distillation.

The package exposes:

* :class:`sprkd.optimizer.SPRKD` - the custom optimizer that performs saddle
  detection (teacher mode) or ASR-targeted descent + Negative Hessian
  Eigensteps + Perturbed Gradient Descent (student mode).
* :func:`sprkd.saddle.is_strong_saddle_point` - the eigenvalue-density saddle
  detection criterion used to populate the saddle-point repository.
* :func:`sprkd.saddle.aggregate_asr` - the Approximated Saddle Region (ASR)
  aggregation step.
* :func:`sprkd.tli.simple_inject` - shape-aware Transfer Learning by Injection
  that pairs same-named layers and injects via center-crop / pad.
* :class:`sprkd.models.MalariaTeacherCNN` and
  :class:`sprkd.models.MalariaStudentCNN` - the two CNN architectures used
  in Experiment 1 of the paper.
* :func:`sprkd.training.train_teacher`, :func:`sprkd.training.train_student`,
  :func:`sprkd.training.train_control` - high-level reproduction utilities.

Reference
---------
Dewan, A., Yogeswaran, A., Fedoruk, B. *SPRKD: Effective Knowledge
Distillation for Deep Neural Networks via Saddle Region Approximation.*
2024.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("sprkd")
except PackageNotFoundError:  # pragma: no cover - fallback when not installed
    __version__ = "0.1.0"

from sprkd.optimizer import SPRKD
from sprkd.saddle import (
    aggregate_asr,
    is_strong_saddle_point,
    SaddlePointRepository,
)
from sprkd.tli import (
    fn_inject,
    pair_layers,
    simple_inject,
)
from sprkd.models import MalariaStudentCNN, MalariaTeacherCNN
from sprkd.architectures import (
    BasicBlock,
    ResNetCIFAR,
    build_resnet20,
    build_resnet32,
    build_resnet44,
    build_resnet56,
)
from sprkd.utils import get_device, set_seed
from sprkd.legacy import (
    enable_legacy_unpickling,
    epoch_validation_series,
    load_legacy_checkpoint,
    load_legacy_metrics_pkl,
    load_legacy_student,
    load_legacy_teacher,
)
from sprkd.stats import McNemarResult, mcnemar_table, mcnemar_test, pairwise_mcnemar
from sprkd.eval import (
    TrialResult,
    collect_predictions,
    evaluate_on_testset,
    evaluate_performance_trials,
)

__all__ = [
    "__version__",
    "SPRKD",
    "aggregate_asr",
    "is_strong_saddle_point",
    "SaddlePointRepository",
    "fn_inject",
    "pair_layers",
    "simple_inject",
    "MalariaStudentCNN",
    "MalariaTeacherCNN",
    "BasicBlock",
    "ResNetCIFAR",
    "build_resnet20",
    "build_resnet32",
    "build_resnet44",
    "build_resnet56",
    "get_device",
    "set_seed",
    "enable_legacy_unpickling",
    "epoch_validation_series",
    "load_legacy_checkpoint",
    "load_legacy_metrics_pkl",
    "load_legacy_student",
    "load_legacy_teacher",
    "McNemarResult",
    "mcnemar_table",
    "mcnemar_test",
    "pairwise_mcnemar",
    "TrialResult",
    "collect_predictions",
    "evaluate_on_testset",
    "evaluate_performance_trials",
]
