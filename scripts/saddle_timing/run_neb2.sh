#!/bin/bash
# Second NEB launch: pair 2:3 (rerun with step clipping and lr 0.01 after the first attempt diverged) and pair 1:2.
cd "$(dirname "$0")"
echo "neb2 start $(date)" >> logs/neb_chain.log
conda run -n sprkd --no-capture-output python neb_ridge_saddle.py --pairs 2:3,1:2 --nodes 12 --neb-images 4096 --hess-images 1024 --string-iters 300 --climb-iters 300 --lr 0.01 --max-step 0.5 --refine-max-steps 60 --student-epochs 30 --student-seeds 0,1,2 > logs/neb_ridge_saddle_2.log 2>&1
echo "neb2 finished $(date) rc=$?" >> logs/neb_chain.log
