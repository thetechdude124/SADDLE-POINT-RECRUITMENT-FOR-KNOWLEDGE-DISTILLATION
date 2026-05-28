# SPRKD: Saddle Point Recruitment for Knowledge Distillation

**Effective Knowledge Distillation for Deep Neural Networks via Saddle
Region Approximation**

> Aditya Dewan (CMU) - Arjun Yogeswaran (Waymo) - Benjamin Fedoruk (Ontario Tech)

Originally awarded **2nd Place, Mathematical and Cybersecurity Research**
(NSA) and **4th Place, Robotics and Intelligent Machines** (Regeneron) at
**Regeneron ISEF 2023**, and selected by **Youth Science Canada** as one of
8 projects to represent Canada in Dallas, TX.

This repository is the official reference implementation accompanying the
paper. The full PDF is at [`docs/sprkd_paper.pdf`](docs/sprkd_paper.pdf). 


(Some agentic AI tools were used to make this existing repository usable for the public ahead of the preprint release, but were rigourously tested)

---

## TL;DR

Standard knowledge distillation (KD) caps a student's accuracy at the
teacher's; SPRKD does not.

SPRKD reframes distillation from *replicating teacher outputs* to *recruiting
teacher saddle points*. Concretely, SPRKD:

1. Trains a (possibly weak) teacher ensemble while monitoring the loss
   landscape via efficient Hessian eigenvalue approximation.
2. Aggregates qualifying low-loss saddle points into an Approximated Saddle
   Region (ASR) and re-parameterises it into the student via Transfer
   Learning by Injection (TLI).
3. Drives the student into the ASR with an exponentially decaying Euclidean
   transformation, then descends via Negative Hessian Eigensteps (NHE) and
   Gaussian Perturbed Gradient Descent (PGD).

On a 6,430-parameter CNN distilled from a **weak** 25,546-parameter teacher
on the NIH/NLM malaria blood smear dataset, SPRKD reaches **94.8% validation
accuracy**, beating Response KD by **+24.70% (McNemar p = 6.3 x 10^-87)** and
matching a strong, scratch-trained control of the same architecture (p = 1.0)
- without ever distilling from a strong teacher.

| Model            | Params  | Val Loss | Val Acc | Err  |
| ---------------- | ------- | -------- | ------- | ---- |
| Control-Teacher  | 25,546  | 0.364    | 94.50   | 5.50 |
| Teacher (Weak)   | 25,546  | 0.583    | 70.13   | 29.87 |
| Control-Student  | 6,430   | 0.364    | 94.47   | 5.53 |
| RKD              | 6,430   | ~0       | 70.10   | 29.90 |
| **SPRKD**        | **6,430** | **0.361** | **94.80** | **5.20** |

---

## Installation

### From source (recommended for now)

```bash
git clone https://github.com/thetechdude124/SADDLE-POINT-RECRUITMENT-FOR-KNOWLEDGE-DISTILLATION.git
cd SADDLE-POINT-RECRUITMENT-FOR-KNOWLEDGE-DISTILLATION

# pull binary checkpoints + saddle points (~300 MB)
git lfs install
git lfs pull

# create a fresh env
conda create -n sprkd python=3.10 -y
conda activate sprkd

# install the package + dev tooling
pip install -e ".[dev]"
```

The package supports CPU, CUDA, and Apple Silicon (MPS) out of the box.

### Optional extras

| Extra        | Purpose                                                                    |
| ------------ | -------------------------------------------------------------------------- |
| `[fastai]`   | `fastai` (used by the original ISEF notebooks for `Learner`s).             |
| `[stats]`    | `statsmodels` so `mcnemar_test(use_statsmodels=True)` matches the canonical Colab. |
| `[tli-full]` | Cross-architecture TLI via `karateclub` / `networkx` / `timm`.              |
| `[legacy]`   | `fastai` + the original `hessian-eigenthings` fork - required only to       |
|              | execute the unmodified `notebooks/SPRKD.ipynb`.                              |
| `[dev]`      | `pytest`, `pytest-cov`, `jupyter`, `nbconvert`, `statsmodels`.              |
| `[all]`      | Everything above (sans `legacy`).                                            |

```bash
pip install -e ".[all]"             # everything except the legacy fork
pip install -e ".[legacy]"          # to run the unmodified ISEF 2023 notebook
```

---

## Quick start

```python
import torch
import torch.nn as nn

from sprkd import (
    MalariaTeacherCNN, MalariaStudentCNN, set_seed,
    aggregate_asr,
)
from sprkd.tli import inject_state_list
from sprkd.training import train_teacher, train_student
from sprkd.data import MalariaDataConfig, make_dataloaders, find_default_root

set_seed(0)
cfg = MalariaDataConfig(root=find_default_root())
train_loader, valid_loader, _ = make_dataloaders(cfg)

# 1. train a weak teacher with saddle tracking
teacher = MalariaTeacherCNN()
sprkd_t, _ = train_teacher(
    teacher, train_loader, valid_loader,
    loss_fn=nn.CrossEntropyLoss(), n_epochs=2, saddle_steps=1,
)

# 2. build the ASR + inject into the student (TLI via teacher shapes)
asr = aggregate_asr([sprkd_t.saddle_repository.snapshots])
student = MalariaStudentCNN()
inject_state_list(student, asr, teacher=teacher)
targets = [p.detach().clone() for p in student.parameters()]

# 3. run the SPRKD student training loop
sprkd_s, history = train_student(
    student, train_loader, valid_loader,
    loss_fn=nn.CrossEntropyLoss(),
    teacher_saddle_points=targets,
    n_epochs=10,
)
print("best val acc:", history.best_valid_acc())
```

A self-contained script lives at
[`examples/quickstart.py`](examples/quickstart.py); larger reproductions live
under [`scripts/`](scripts).

---

## Command-line interface

```bash
sprkd info                                         # env / model summary
sprkd train-teacher  --epochs 2 --output checkpoints/teacher.pth
sprkd build-asr      --teachers checkpoints/teacher.pth --output checkpoints/asr.pth
sprkd train-student  --asr checkpoints/asr.pth --epochs 10 --output checkpoints/student.pth
sprkd eval           --student checkpoints/student.pth
```

Each subcommand accepts `--data-root /path/to/cell_images`, `--batch-size`,
`--num-workers`, `--seed`, and `--quiet`.

---

## Project layout

```
SADDLE-POINT-RECRUITMENT-FOR-KNOWLEDGE-DISTILLATION/
|-- sprkd/                  # the pip-installable package
|   |-- optimizer.py        # SPRKD optimizer (teacher / control / student)
|   |-- saddle.py           # strong-saddle detection (3 paper-traceable rules)
|   |-- tli.py              # Transfer Learning by Injection (lightweight)
|   |-- training.py         # train_teacher / train_student / train_control / RKD
|   |-- models.py           # MalariaTeacherCNN, MalariaStudentCNN (paper Sec. 4.1)
|   |-- architectures.py    # CIFAR-style ResNet-20/32/44/56 (paper Sec. 4.2)
|   |-- data.py             # malaria + CIFAR-100 + MNIST + TinyImageNet loaders
|   |-- analysis.py         # Hessian trace / ESD / spectrum
|   |-- landscape.py        # 2-D loss landscape via top-2 eigenvectors (Fig. 4)
|   |-- stats.py            # McNemar test (in-house + statsmodels-backed)
|   |-- eval.py             # trial-averaged evaluation utilities
|   |-- visualize.py        # paper figures
|   |-- legacy.py           # backward compat for the ISEF 2023 .pth artifacts
|   `-- cli.py              # `sprkd ...` command-line interface
|-- tests/                  # 186 unit + integration tests
|-- scripts/                # reproduction scripts (see scripts/README.md)
|-- examples/               # quickstart.py
|-- notebooks/              # SPRKD.ipynb (original) + SPRKD_quickstart.ipynb
|                           # + SPRKD_data_analysis.ipynb (paper Figs. 2-4 + Tab. 1)
|-- docs/                   # paper PDF, LaTeX source, ALGORITHM.md
|-- MODELS/                 # released student / teacher / ASR checkpoints (LFS)
|-- METRICS/                # per-step training metrics + Hessian eigenspectra
|-- cell_images/            # NIH/NLM malaria dataset (Parasitized / Uninfected)
|-- *_SADDLE_POINTS.pth     # released teacher saddle-point repositories (LFS)
|-- pyproject.toml          # package metadata
|-- LICENSE                 # MIT
|-- CITATION.cff            # citation metadata
`-- CHANGELOG.md
```

---

## Reproducing the paper

The released model checkpoints are stored as **git-lfs** artifacts. They
must be pulled before reproducing or evaluating any results.

```bash
git lfs install
git lfs pull
```

Then:

```bash
# verify all tests (~45s on Apple Silicon; faster on GPU machines)
pytest -q

# run the full reproduction (Experiment 1, malaria blood smear)
python scripts/reproduce_malaria.py --data-root cell_images --epochs 10

# evaluate the released SPRKD student on the malaria validation split
python scripts/eval_released_student.py
```

`scripts/eval_released_student.py` loads `MODELS/SPRKD_MALARIA.pth` via the
backward-compat `sprkd.legacy.load_legacy_student` helper, which re-creates
the legacy fastai `Learner` pickle into a clean `MalariaStudentCNN`.

---

## Provenance and paper-faithful defaults

The package ports the **canonical Colab implementations** of the SPRKD paper
unchanged. Two source notebooks dictate every default in the package:

| Source notebook (Colab)                                                | Used for                                |
| ---------------------------------------------------------------------- | --------------------------------------- |
| `SPRKD_SADDLE_POINT_RECRUITMENT_FOR_KNOWLEDGE_DISTILLATION_ADITYA_DEWAN_2023` | Malaria experiment (paper Sec. 4.1)     |
| `EXPERIMENTAL_MODEL_EVALUATIONS`                                       | TinyImageNet experiment (paper Sec. 4.2) |

Specifically, the package matches the canonical Colab for:

| Component                                | Default behaviour                            | Source                                |
| ---------------------------------------- | -------------------------------------------- | ------------------------------------- |
| Strong-saddle rule                       | `\|sum(neg eigs)\| >= 7` (`rule="magnitude"`) | latest Colab cell `determineSaddlePoint` |
| Negative Hessian Eigenstep weight        | `1 / \|lambda_neg\|` (`nhe_step_mode="adaptive"`) | latest Colab cell `negativeHessianEigensteps` |
| PGD perturbation                         | `xi ~ N(0, 0.1)` (`pgd_perturb_variance=0.1`) | paper Sec. 3.3.2 + Colab `perturbedGD`  |
| `epsilon` (TM termination)               | `1e-3`                                       | latest Colab `train_student` call       |
| `pgd_grad_threshold`                     | `0.01`                                       | latest Colab `train_student` call       |
| McNemar test (`use_statsmodels=True`)    | delegates to `statsmodels.stats.contingency_tables.mcnemar` | `EXPERIMENTAL_MODEL_EVALUATIONS` cell  |

The legacy alpha-ratio rule from the original ISEF notebook is still
available via `SaddleCriterion(rule="ratio", alpha=0.4)`, and the literal
paper Equation 1 (logical AND of both conditions) via `rule="both"`.

---

## Loading legacy ISEF 2023 checkpoints

The original notebook saved fastai `Learner` objects whose `opt` slot
referenced a notebook-level `SPRKD` class living in `__main__`. Re-loading
them in a fresh Python process therefore raises
`AttributeError: Can't get attribute 'SPRKD'`. This package ships
`sprkd.legacy` to handle the situation transparently:

```python
from sprkd import load_legacy_student
model = load_legacy_student("MODELS/SPRKD_MALARIA.pth")
model.eval()
```

`sprkd.legacy.enable_legacy_unpickling()` registers minimal stub classes in
`__main__` so any `torch.load(..., weights_only=False)` call from any user
script will succeed.

See [`tests/test_checkpoints.py`](tests/test_checkpoints.py) for further
examples.

---

## Tests

```bash
pytest -q                          # 186 tests
pytest -m paper                    # sprkd.tex + bundled artifact checks
pytest -m checkpoints              # LFS-dependent checkpoint tests
pytest -k "saddle or tli" -v       # specific subsets
```

The suite covers:

| Area               | Tests                                                       |
| ------------------ | ----------------------------------------------------------- |
| Imports / packaging| `tests/test_imports.py`                                     |
| Models             | `tests/test_models.py`  (param counts match paper Table 1)   |
| Saddle detection   | `tests/test_saddle.py`  (strong/weak/edge cases + ASR avg)   |
| TLI                | `tests/test_tli.py`     (`fn_inject`, layer pairing, etc.)   |
| Optimizer          | `tests/test_optimizer.py` (teacher/control/student modes)   |
| Training           | `tests/test_training.py` + `tests/test_integration.py`      |
| Data               | `tests/test_data.py`    (synthetic + real malaria split)    |
| Analysis           | `tests/test_analysis.py`                                    |
| Visualization      | `tests/test_visualize.py`                                   |
| Utils              | `tests/test_utils.py`                                       |
| CLI                | `tests/test_cli.py`                                         |
| Legacy             | `tests/test_legacy.py` + `tests/test_checkpoints.py`        |
| Paper / perf       | `tests/test_paper.py` + `tests/test_performance.py`         |

**Install name:** `pip install -e .` installs the **`sprkd`** package (`import sprkd`).

---

## Citing this work

If you use SPRKD in your research, please cite:

```bibtex
@article{dewan2024sprkd,
  title   = {SPRKD: Effective Knowledge Distillation for Deep Neural Networks via Saddle Region Approximation},
  author  = {Dewan, Aditya and Yogeswaran, Arjun and Fedoruk, Benjamin},
  year    = {2024},
}
```

A `CITATION.cff` is shipped alongside this README for GitHub's automatic
citation widget.

---

## Acknowledgements

We thank the organizers of the SciComm Viewpoint Challenge for the initial
intellectual context, the Team Canada ISEF delegation for support during
competition, and reviewers of earlier versions of this work for substantive
feedback.

This package vendors a small portion of Maciej Czyzewski's `tli-pytorch`
(MIT) for the cross-architecture TLI path used in Experiment 2; see
[`LICENSE`](LICENSE) for full attribution.

---

## License

MIT - see [`LICENSE`](LICENSE).

The accompanying paper (`docs/sprkd_paper.pdf`) is licensed CC BY 4.0 per
the arXiv submission metadata.
