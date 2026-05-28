# Notebooks

## `SPRKD.ipynb`

This is the **original ISEF 2023 notebook**, preserved unmodified. It
contains the in-line implementation of the SPRKD optimizer, the vendored
copy of the original ``tli-pytorch`` graph-matching code, and the full
teacher-ensemble / student training pipeline as it was at the time of the
2023 ISEF submission.

The notebook is included for archival fidelity; it is **not** the
recommended entry point for new work. Two reasons:

1. The very first cell reinstalls a numpy / hessian-eigenthings / pyhessian
   matrix that conflicts with modern Python environments.
2. Many cells assume CUDA is available (`torch.cuda.set_device(0)`, etc.).

### Running the legacy notebook

If you genuinely need to execute the legacy notebook end-to-end:

```bash
conda activate sprkd
pip install -e ".[fastai,tli-full]"
jupyter lab notebooks/SPRKD.ipynb
```

You may need to comment out the very first install cell. Skip the
`torch.cuda.*` lines on machines without CUDA.

### Recommended replacement: the package APIs

```python
from sprkd import MalariaTeacherCNN, MalariaStudentCNN, aggregate_asr
from sprkd.tli import inject_state_list
from sprkd.training import train_teacher, train_student
from sprkd.data import MalariaDataConfig, find_default_root, make_dataloaders
```

The package equivalents replicate every piece of logic in `SPRKD.ipynb`
- including legacy checkpoint loading via ``sprkd.legacy.load_legacy_*``
- and run unchanged on CPU / CUDA / Apple Silicon (MPS).

A minimal, self-contained, executable demo lives at
[`examples/quickstart.py`](../examples/quickstart.py).
