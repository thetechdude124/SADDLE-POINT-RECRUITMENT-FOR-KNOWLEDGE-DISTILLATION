# SPRKD: Algorithm Overview

This document is a quick-reference summary of the SPRKD pipeline. The full
treatment is in [`sprkd_paper.pdf`](./sprkd_paper.pdf), Sections 2-3.

## Notation

| Symbol           | Meaning                                                    |
|------------------|-----------------------------------------------------------|
| `theta`          | Model parameters                                          |
| `H_f(theta)`     | Hessian of the loss `f` at `theta`                         |
| `lambda_i`       | i-th Hessian eigenvalue                                   |
| `v`              | Eigenvector of the largest-magnitude negative eigenvalue  |
| `eta`            | Inner-optimizer learning rate                             |
| `epsilon`        | Distance threshold for ASR convergence                    |
| `alpha`          | Saddle-detection ratio threshold (default 0.4)            |
| `beta`           | Saddle-detection magnitude threshold (default 7)          |
| `j`              | PGD gradient-norm threshold (default 0.02)                |

## Phase 1 - Teacher ensemble training

For each teacher `T_k`, k = 1...K:

```
for each step t in {1, ..., N_teacher_steps}:
    base_optimizer.step()                                # standard Adam step

    if t mod saddle_steps == 0:
        eigenvalues = top_n(H_f(theta_t), n_top_eigs)    # Lanczos / power iter
        if is_strong_saddle_point(eigenvalues, alpha, beta):
            repository.append(theta_t)                   # CPU snapshot
```

Cost: ~1 backprop per saddle check (Hessian-vector products only; never the
full Hessian).

## Phase 2 - ASR construction

```
ASR = mean_k( best_loss_snapshot(repository[T_k]) )      # one tensor / layer
student.parameters() <- TLI.simple_inject(ASR, into=teacher_arch).reproject()
```

`simple_inject` reduces, layer-by-layer, to a center-aligned crop / pad so
matched-depth architectures (Experiment 1) avoid the full graph-matching
machinery of the Czyzewski TLI implementation.

## Phase 3 - Student training

Three sub-phases driven by the SPRKD optimizer:

### 3a. Iterative ASR approaching (Transformation Matrix)

```
for each parameter S_i and target T_i in ASR:
    M_i = T_i / S_i                                      # element-wise
    weight = -2^{-t/10} / 2 + 1                          # decay -> 1
    S_i <- S_i * (weight * M_i)
    if max_diag(euclidean_distance(S_i, T_i)) <= epsilon:
        disable_targeting(i)
```

Continues for 10-200 steps depending on layer depth/width (paper Section 3.3.1).

### 3b. Negative Hessian Eigensteps (NHE)

After ASR is reached:

```
if grad_norm(S) <= j and not in_cooldown:
    lambda_neg, v = top_negative_eigenpair(H_f(theta_t))
    theta_{t+1} = theta_t - 0.5 * H_f(theta_t) * v * v
```

NHE traverses curated negative-curvature directions, accelerating descent
into minima clusters below the visited saddles.

### 3c. Gaussian Perturbed Gradient Descent (PGD)

Following each NHE attempt:

```
xi ~ N(0, sigma^2 * I)                                   # sigma = 0.1
theta_{t+1} = theta_t - xi
if loss(theta_{t+1}) > loss(theta_t):
    revert()                                             # reject move
```

PGD escapes near-degenerate saddles within polynomial time (Jin et al. 2017).

## Convergence properties

The combined SPRKD loop converges to second-order stationary points
(non-saddles) in time matching SGD's first-order stationary point
convergence, independent of model dimensionality. See paper Section 3.3.2.
