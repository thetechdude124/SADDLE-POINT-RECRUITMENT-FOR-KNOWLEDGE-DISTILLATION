# Paper vs. code reconciliation

Every hyperparameter and mechanism of SPRKD, with the value in the arXiv v1 text
(`sprkd.tex`), in the ISEF notebook that produced the released numbers
(`notebooks/SPRKD.ipynb`, "notebook"), in the released package before the September 2026
fixes (`sprkd` 0.1.0, "package v0.1"), and the default the fixed package uses now
(`sprkd` 0.2.0). "Final cell" means the last `train_student` call in
the notebook (cell 23), which is the configuration of the saved run.

| item | paper | notebook | package v0.1 | package v0.2 (now) | note |
|---|---|---|---|---|---|
| model output | softmax probabilities (Fig. 2 arch.) | `nn.Softmax(dim=1)` then `CrossEntropyLoss` | same | **logits**; softmax lives in the loss | double softmax removed; checkpoints load unchanged |
| saddle rule | Eq. 1: `\|sum neg\| > beta` **and** `sum pos > alpha sum \|neg\|` (alpha 0.4, beta 7) | ratio only: `\|sum neg\| >= 0.4 sum pos` (cell 4) | magnitude only: `\|sum neg\| >= 7` | magnitude only (default), `ratio` and `both` selectable; **optional gradient-norm gate** `max_grad_norm` (off by default) | the paper's Eq. 1 second inequality is written with the sides swapped relative to the notebook; neither run used both |
| gradient check at a "saddle" | implied by the definition (grad = 0) | none | none | none by default; gate available; grad norm logged per snapshot | |
| eigenvalues checked | top z = 4 (student), 2-20 (teacher) | `n_eigs = 2` in `determineSaddlePoint` | `n_top_eigs = 4` | 4 | |
| check stride `k` | "k in [1, 50]" | `saddle_steps = 1` | 50 (`train_teacher` passes 1) | 50 (`train_teacher` still passes 1 for compatibility; the harness uses 50-200) | stride 1 is 100-800 extra HVP passes per step |
| snapshot retention | all qualifying points | all | all | all, or top-k by loss (`saddle_top_k`) | |
| ASR aggregation | average of the lowest-loss saddle per teacher | last snapshot of teacher 1, divided by K (out-of-place `add`, cell 20) | mean of the **last** snapshot per teacher | mean of the **lowest-loss** snapshot per teacher (`select="best"`) | the notebook's ensemble result was a single 1/3-scaled checkpoint |
| injection | TLI graph matching (Czyzewski) | upstream `tli-pytorch` `apply_tli` | key matching + center crop, equal parameter counts only; `transfer_via_graph` imported a missing module | key matching + center crop/pad, depth-mismatched and smaller-teacher pairs allowed; `transfer_via_graph` raises `NotImplementedError` | |
| student start | "not direct initialisation": random init pulled toward the ASR | fresh random student, TLI tensors as targets | student initialised **at** the injected tensors and targets set equal, so the TM phase ends at step 1 | unchanged in the package; the harness (`neurips/bench/sprkd_runner.py`) restores the notebook semantics (random start, injected targets) and offers `--no-tm` | |
| decay `l` | `l = 1 - 2^(-t/10)` | `1 - 2^(-t/10) / 2` (starts at 0.5) | same as notebook | same as notebook | paper text omits the /2; the paper's form would zero the student at t = 0 |
| TM update | `S' = S (l M)`, `M = T / S` | `param.squeeze().mul(weight * T/param)` = `weight * T` | `p * (weight * T / p)` = `weight * T` | same | the division cancels: TM is an overwrite with a scaled copy of the target |
| SGD during TM | not stated | Adam step skipped while any tensor is targeting | same | same | |
| `epsilon` (TM stop) | 0.1 | `10e-4` = 1e-3 (final cell) | 1e-3 | 1e-3 | paper text should read 1e-3 |
| distance metric | `max_{rc, r=c} sqrt(lambda_max((S'-T)^T (S'-T)))` | diagonal of `torch.cdist` (double diagonal for rank > 2) | same | same | not the stated formula; kept for reproducibility, flagged for the rewrite |
| stagnation threshold `j` | 0.02 | 0.02 (final cell); 0.01 in an intermediate cell | 0.01 | **0.02** | |
| PGD distance gate `PGD_delta` | not in the paper | 0.25 (final cell); 5 in an earlier cell | 5.0 | **0.25** | with the student starting at the target the gap is ~0; 5.0 never triggered |
| PGD epoch limit | not in the paper | 15 (final cell); 100 earlier | 100 | 100 | |
| PGD loss-progress condition | not in the paper | perturb only if loss fell >= 0.002 since the last event | same | same | |
| perturbation | `xi ~ N(0, 0.1)` | radius `(j / (8 lambda_max)) sqrt(pi/6430)`, std = sqrt(radius) ~ 1e-3 | `sqrt(0.1) * N(0,1)` on the flagged tensor | `sqrt(0.1) * N(0,1)` on the flagged tensor (variance configurable; 0 disables) | paper and notebook disagree by two orders of magnitude |
| NHE step | `theta -= (1/\|lambda\|) (grad * v * v)` (elementwise) | commented out in `step()` and in `perturbedGD()`; the function body used `0.1 * grad * v * v` | `(1/\|lambda\|) grad * v * v`, but grads were None after the Hessian call, so no parameter changed | **`theta -= eta * sign(g.v) * v`** for every negative top eigenpair, `eta = min(0.1, \|lambda\|)` (adaptive) or 0.1 (fixed) | the released numbers were produced without any NHE |
| NHE eigenpairs | largest-magnitude negative | `n_eigs = 2` | top 2 | top 2 (`n_nhe_eigs`) | |
| NHE cap | not stated | 1000 (final cell) | 50 | 50 (`max_nhe_steps`; 0 disables) | |
| revert | "if NHE + perturbation jointly fail to decrease loss, revert" | commented out | history stored, never used | **implemented**: NHE alone reverts on loss increase; the NHE + PGD event reverts as a whole if the batch loss did not fall | |
| cooldown | not stated | 10 (final cell) | 20 | 20 | |
| teacher optimiser | "SGD" in Sec. 1 text; Adam in practice | Adam 1e-3 | Adam 1e-3 | Adam 1e-3 | |
| teacher budget | "2 epochs" | 2 epochs x 700 samples (~1,400 images) | `n_epochs` full passes | same | the paper's "2 epochs" is ~7% of two real epochs |
| student budget | 500 epochs x 323 steps | 10 epochs x 700 samples (final cell; the 500-epoch logs come from other runs) | user-set | user-set | |
| Response KD loss | "standard logit matching" (KL) | not in this notebook | `KL(log_softmax(s), softmax(t))`, T = 1, no CE, on softmax outputs | **`alpha CE + (1-alpha) T^2 KL`** on logits, alpha 0.5, T 4 (configurable) | |
| McNemar | exact, on the 6,890-sample validation split | `statsmodels.mcnemar` in the evaluation notebook | in-house exact + statsmodels option | unchanged; the harness adds paired McNemar on a fixed held-out test set and a TOST equivalence test | the v1 test paired models trained on different random splits |
| "5 trials" | 5 training trials averaged | 5 evaluation resamples of 100 images (`evaluate_performance_trials`) | same | unchanged; the harness runs 3 training seeds | |
| Hessian device | Colab GPU | GPU | CPU on Apple silicon (PyHessian has no MPS path) | same | |
| model mode during Hessian | n/a | left in eval mode afterwards | same | **restored** (train/eval mode and gradients) | |

## Values the fixed package uses by default

`SPRKD(...)`: `saddle_steps=50`, `n_top_eigs=4`, `SaddleCriterion(rule="magnitude",
alpha=0.4, magnitude_threshold=7.0, max_grad_norm=None)`, `saddle_top_k=None`,
`epsilon=1e-3`, `pgd_grad_threshold=0.02`, `pgd_delta=0.25`, `pgd_epoch_limit=100`,
`pgd_perturb_variance=0.1`, `max_nhe_steps=50`, `cooldown_steps=20`,
`nhe_step_mode="adaptive"`, `nhe_step_size=0.1`, `n_nhe_eigs=2`, `revert_on_increase=True`.

`train_response_kd(...)`: `alpha=0.5`, `temperature=4.0`, Adam `lr=1e-3`, same epochs as
the other arms.

## What the paper text should say after the rewrite

- Eq. 1: state the rule actually used (one of the three), plus the gradient-norm gate if
  adopted, and report how many iterates qualified.
- Section 3.3.1: `l = 1 - 2^(-t/10)/2`, `epsilon = 1e-3`, and the honest description that
  the phase sets `S = l T` with the base optimiser paused, i.e. a delayed initialisation.
- Section 3.3.2: the corrected NHE step, the actual perturbation scale, the revert, and the
  event counts per run.
- Section 4: the KD baseline loss, teacher budget in images, seeds, split protocol.
