# Changelog

All notable changes to the SPRKD package are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-25 (corrections)

### Fixed
- `MalariaTeacherCNN` / `MalariaStudentCNN` return logits; the trailing `nn.Softmax`
  meant `CrossEntropyLoss` saw a double softmax. Checkpoints load unchanged.
- `train_response_kd` uses the standard Hinton loss
  `alpha * CE + (1 - alpha) * T^2 * KL` on logits (alpha 0.5, T 4 by default). The
  previous loss had no label term and was applied to softmax outputs at T = 1, so the
  arXiv v1 Response KD comparison (24.7 points) is not a valid comparison.
- All PyHessian calls preserve train/eval mode and gradients (`sprkd.hessian_utils`).
  Previously teachers ran with dropout disabled after the first saddle check of each
  epoch and the NHE step never modified a parameter.
- Negative Hessian Eigenstep is a negative-curvature step
  `theta -= eta * sign(g.v) * v` with revert on loss increase; the NHE + PGD event is
  reverted as a whole when it fails to lower the batch loss. Event counters via
  `SPRKD.counters()`.
- `aggregate_asr` averages each teacher's lowest-loss snapshot (accepts
  `SaddlePointRepository` objects); `SaddlePointRepository` records loss, gradient
  norm, step and rule outcome, and supports top-k retention.
- `SaddleCriterion.max_grad_norm` gradient-norm gate (off by default).
- `inject_state_list` supports depth-mismatched pairs and a teacher smaller than the
  student when a teacher module is supplied.
- TinyImageNet validation images are normalised like the training images.
- `transfer_via_graph` raises `NotImplementedError` instead of importing a module that
  is not shipped.

### Changed
- Defaults aligned with the paper and the notebook's final run: `pgd_grad_threshold`
  0.02 (was 0.01), `pgd_delta` 0.25 (was 5.0).
- README gains an "Errata and corrections" section. The CIFAR-100 and MNIST numbers in
  the v1 text have no code in this repository.

### Added
- `tests/test_fixes.py` (regression tests, one per correction).

## [0.1.0] - 2026 (paper release)

### Added
- Initial pip-installable layout (`pyproject.toml`, `sprkd/` package).
- Public modules:
  - `sprkd.optimizer.SPRKD` - the SPRKD optimizer (teacher / control / student modes).
  - `sprkd.saddle` - strong-saddle detection (three paper-traceable rules:
    `"magnitude"` (default; matches the canonical Colab),
    `"ratio"` (matches the original ISEF notebook), and `"both"` (paper Eq. 1
    read literally)).
  - `sprkd.tli` - shape-aware Transfer Learning by Injection (`fn_inject`,
    `simple_inject`, `pair_layers`, `inject_state_list`).
  - `sprkd.training` - high-level loops for teacher / student / control /
    Response KD training.
  - `sprkd.models` - `MalariaTeacherCNN` (25,546 params), `MalariaStudentCNN`
    (6,430 params), and `build_legacy_sequential_*` reproductions.
  - `sprkd.architectures` - CIFAR-style `ResNetCIFAR`, `BasicBlock`, and
    `build_resnet20/32/44/56` for paper Section 4.2 (TinyImageNet).
  - `sprkd.data` - malaria, CIFAR-100, MNIST, and TinyImageNet (with
    val-folder reorganisation) DataLoader utilities.
  - `sprkd.analysis` - PyHessian wrappers (trace, ESD, top eigenvalue).
  - `sprkd.landscape` - 2-D loss landscape sweep along the top-2 Hessian
    eigenvectors (paper Figure 4).
  - `sprkd.stats` - McNemar paired test (in-house exact + chi-squared, with
    optional `statsmodels` delegation matching the canonical Colab).
  - `sprkd.eval` - trial-averaged evaluation utilities mirroring
    `evaluate_performance_trials` from `EXPERIMENTAL_MODEL_EVALUATIONS.ipynb`.
  - `sprkd.visualize` - paper-ready loss/accuracy, ESD, and 3-D landscape plots.
  - `sprkd.legacy` - backward-compatible loaders for the original ISEF 2023
    notebook checkpoints (model + optimizer + metric .pkl extraction).
  - `sprkd.cli` - `sprkd info / train-teacher / build-asr / train-student / eval`.
- 160+ unit/integration tests covering all of the above.
- `notebooks/SPRKD_quickstart.ipynb` (modern quickstart, executable on MPS).
- `notebooks/SPRKD_data_analysis.ipynb` (loss/acc, ESD, McNemar, landscape).
- `sprkd.legacy.load_legacy_metrics_pkl` and `epoch_validation_series` for CPU-safe
  loading of CUDA-pickled `METRICS/LOSSES AND ACCURACIES/*.pkl` files.
- `tests/test_paper.py` and `tests/test_performance.py` (paper + timing smoke tests).
- Executed `notebooks/SPRKD_data_analysis_executed.ipynb`.
- LaTeX paper, ALGORITHM.md, CITATION.cff, LICENSE.

### Defaults reconciled with the canonical SPRKD Colab notebooks

The optimizer's defaults exactly match
`SPRKD_SADDLE_POINT_RECRUITMENT_FOR_KNOWLEDGE_DISTILLATION_ADITYA_DEWAN_2023`
(latest version) and `EXPERIMENTAL_MODEL_EVALUATIONS`:

- Saddle criterion: `|sum(neg eigs)| >= 7` (default `rule="magnitude"`).
- NHE weight: `1 / |lambda_neg|` (default `nhe_step_mode="adaptive"`).
- PGD perturbation: `xi ~ N(0, 0.1)` (paper Sec. 3.3.2;
  `pgd_perturb_variance=0.1`).
- New `saddle_step_limit` argument matches the corresponding parameter in the
  TinyImageNet Colab.
- New `[stats]` extra installs `statsmodels` so
  `mcnemar_test(use_statsmodels=True)` matches the canonical Colab byte-for-byte.

### Notes
- Released as the official artifact for the arXiv submission of *SPRKD:
  Effective Knowledge Distillation for Deep Neural Networks via Saddle Region
  Approximation*.
- Original ISEF 2023 notebook is preserved unmodified at
  `notebooks/SPRKD.ipynb` and remains executable against this package via
  the legacy loaders. The ratio-based saddle rule used by that notebook is
  available via `SaddleCriterion(rule="ratio", alpha=0.4)`.

## [0.0.0] - 2023 (initial commit)

Comitted basic colab notebooks and saddle points .pth files ahead of Regeneron International Science and Engineering Fair. 