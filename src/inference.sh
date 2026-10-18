#!/bin/sh

python -u src/inference.py  \
    --model    neptune-1    \
    --year     2019         \
    --ensemble 8            \
    --ocean    "oras5"      \
    --atmos    "era5"       \
    --res      "1"
