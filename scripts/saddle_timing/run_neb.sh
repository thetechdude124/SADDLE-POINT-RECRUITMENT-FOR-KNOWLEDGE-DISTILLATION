#!/bin/bash
# Ridge-saddle NEB experiment (malaria, local). nohup ./run_neb.sh > logs/neb_nohup.log 2>&1 &
cd "$(dirname "$0")"
echo "neb start $(date)" >> logs/neb_chain.log
conda run -n sprkd --no-capture-output python neb_ridge_saddle.py --pairs 0:1,2:3 --nodes 12 --neb-images 4096 --hess-images 1024 --string-iters 300 --climb-iters 300 --refine-max-steps 60 --student-epochs 30 --student-seeds 0,1,2 > logs/neb_ridge_saddle.log 2>&1
echo "neb finished $(date) rc=$?" >> logs/neb_chain.log
