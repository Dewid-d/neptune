#!/bin/sh

# NUM_PROCS is the number of concurrent processes that merge and interpolate the data
mpirun python -u src/processing/merge_and_interp_all_tensors_parallel.py \
    --shape '(721,1440)' \
    --dst '025' \
    --mode 'bilinear'
