#!/bin/sh

# spawn concurrent NUM_PROCS processes to download 
# the ERA5 data from weatherbench in parallel
mpirun -np NUM_PROCS python -u weatherbench_daily.py
