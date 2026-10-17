# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# ----------------------------------------------------------------------------
from atmocean.macros import STD_DATA_VARS, MEAN_DATA_VARS

from typing import List, Dict
import xarray as xr
import numpy as np
import logging
import torch
import glob
import os


def unnormalize(x, field, epsilon):
    x = (STD_DATA_VARS[field] + epsilon) * x + MEAN_DATA_VARS[field]
    return x


def normalize(x, field, epsilon):
    x = (x - MEAN_DATA_VARS[field]) / (STD_DATA_VARS[field] + epsilon)
    return x
