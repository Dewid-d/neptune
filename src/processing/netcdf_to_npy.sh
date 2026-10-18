#!/bin/sh

# use the number of processes that you wish to use to concurrently process your data
mpirun -np NUM_PROCS python -u src/processing/netcdf_to_npy.py --domain ocean
