# Reproduction scripts

Each script is a thin wrapper around the public package APIs and is safe to
import (no top-level side effects). Run from the repo root.

| Script                          | Purpose                                                   |
| ------------------------------- | --------------------------------------------------------- |
| `reproduce_malaria.py`          | End-to-end Experiment 1 (teacher ensemble -> ASR -> student). |
| `eval_released_student.py`      | Load `MODELS/SPRKD_MALARIA.pth` and report validation accuracy. |
| `train_teacher_ensemble.py`     | Train an ensemble of weak teachers, save saddles per teacher. |
| `build_asr.py`                  | Aggregate teacher saddles into an ASR `.pth`.              |
| `train_student.py`              | SPRKD student training given an ASR.                      |
| `train_response_kd_baseline.py` | Train the Response KD baseline reported in Table 1.       |

All scripts respect the same flags:

```
--data-root PATH        # default: ./cell_images
--batch-size INT
--num-workers INT
--seed INT
--epochs INT
--device {cpu, cuda, mps}
```
