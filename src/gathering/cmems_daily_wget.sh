#!/bin/sh

# each process will download a month of data
mpirun -np 12 python -u cmems_daily_wget.py --year 2022