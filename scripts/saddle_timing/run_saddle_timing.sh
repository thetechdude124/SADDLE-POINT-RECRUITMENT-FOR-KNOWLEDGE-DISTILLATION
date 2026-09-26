#!/bin/bash
# Saddle-timing study launcher (resumable; completed checkpoints/metrics are skipped).
cd "$(dirname "$0")"
echo "saddle timing study launcher start $(date)" >> logs/saddle_timing_chain.log
conda run -n sprkd --no-capture-output python saddle_timing_study.py --stage all --teacher-seeds 0,1,2 --epochs 40 --student-epochs 30 --student-seeds 0,1,2 --transplant-teacher-seeds 0,1,2 --basin-cont-seeds 4 --control-100-epochs 10 --refine-max-steps 40 --refine-grad-tol 1e-2 >> logs/saddle_timing_study.log 2>&1
echo "saddle timing study finished $(date) rc=$?" >> logs/saddle_timing_chain.log
