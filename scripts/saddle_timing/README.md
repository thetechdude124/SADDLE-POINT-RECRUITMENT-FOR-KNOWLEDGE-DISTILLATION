# Saddle timing study (malaria)

Scripts used for the September 2026 saddle-timing analysis: does the nearest stationary
point of a weak teacher's checkpoint change character along training, do refined points sit
between basins, and do they transplant into the student differently from raw checkpoints?

- `saddle_timing_study.py` - four resumable stages (checkpoints + curvature metrics,
  refinement, basin-choice continuations, transplant).
- `plot_saddle_timing.py` - tables (markdown) and figures from the stage JSON files.
- `run_saddle_timing.sh` - the launcher with the settings that were run.

These scripts were written against the benchmark harness in the paper working folder
(`neurips/bench/`: `common.py`, `malaria_e1.py`, the cached malaria tensors and the fixed
split file) and import from it; copy them next to that harness to run them. They depend on
`sprkd` 0.3.0 (`extreme_eigenpairs(k=...)`, `refine_to_stationary`).
