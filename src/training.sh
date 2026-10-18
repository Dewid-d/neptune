#!/bin/sh

# NOTE
# NUM_NODES is the number of nodes on your system
# NUM_DEVICES_PER_NODE is the number of devices per node available on your system
# NUM_PROCS = NUM_NODES * NUM_DEVICES_PER_NODE

# Pre-Training
mpirun -np NUM_PROCS python -u src/training.py \
    --config src/config/neptune-1.toml \
    --num_nodes NUM_NODES \
    --devices NUM_DEVICES_PER_NODE \
    --ocean oras5 \
    --atmos era5
