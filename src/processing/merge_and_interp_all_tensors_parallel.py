# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# ----------------------------------------------------------------------------

import torch.nn.functional as F
from typing import List
from mpi4py import MPI
import numpy as np
import argparse
import logging
import torch
import munch
import glob
import toml
import sys
import os

MAIN_DIR = 'PATH-TO-PROJECT-DIR'

##########################################################################################
# Custom defined functions
##########################################################################################
def round_robin_split(l: List, n_proc: int):
    """ Split a list into n_proc sublist in a round robin fashion. """
    lists = [[] for _ in range(n_proc)]
    i = 0
    for elem in l:
        lists[i].append(elem)
        i = (i + 1) % n_proc
    return lists
##########################################################################################

# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()


PROGRAM_NAME = r'''Merge all Tensors and Interpolate 0.25° data to coarse resolution provided as input'''
PROGRAM_DESCRIPTION = 'Interpolate to coarse resolution grid'
PROGRAM_ARGUMENTS = [
    [('-s', '--shape'), {'type':str, 'help':'Target (lat,lon) shape', 'required':True}],
    [('-d', '--dst'), {'type':str, 'help':'Destination dir name', 'required':True}],
    [('-m', '--mode'), {'type':str, 'help':'Interpolation mode: bilinear, nearest', 'required':True}],
]

# parse cl arguments
parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
for arg in PROGRAM_ARGUMENTS:
    names, kwargs = arg
    sname, lname = names
    parser.add_argument(sname, lname, **kwargs)
args = parser.parse_args()

input_shape = (721, 1440)
target_shape = eval(args.shape)
target_resolution = args.dst
interpolation_mode = args.mode

dst_dir = os.path.join(MAIN_DIR, f'data/datasets/all_{target_resolution}_deg')
atm_3d_src = os.path.join(MAIN_DIR, 'data/datasets/atmosphere/processed/3D')
atm_2d_src = os.path.join(MAIN_DIR, 'data/datasets/atmosphere/processed/2D')
oce_3d_src = os.path.join(MAIN_DIR, 'data/datasets/ocean/processed/3D')
oce_2d_src = os.path.join(MAIN_DIR, 'data/datasets/ocean/processed/2D')

os.makedirs(dst_dir, exist_ok=True)

log_dir = os.path.join(MAIN_DIR, 'logs/processing/merge_in_single_tensor')

atm_3d_fnames = sorted(glob.glob(os.path.join(atm_3d_src, '*.npy')))
atm_2d_fnames = sorted(glob.glob(os.path.join(atm_2d_src, '*.npy')))
oce_3d_fnames = sorted(glob.glob(os.path.join(oce_3d_src, '*.npy')))
oce_2d_fnames = sorted(glob.glob(os.path.join(oce_2d_src, '*.npy')))

# get fnames
fnames = [os.path.basename(fname) for fname in atm_3d_fnames]

if not rank:
    assert len(atm_3d_fnames) == len(atm_2d_fnames) == len(oce_3d_fnames) == len(oce_2d_fnames)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger 
logging_level = logging.INFO
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s : %(message)s", 
    filename=f"{log_dir}/proc-{rank+1}.log", 
    filemode="w", 
    level=logging_level, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# split the whole filenames list into "n_proc" sublists in a round robin fashion 
fnames_per_proc = round_robin_split(fnames, size)[rank]

# log
logging.info(f'Number of total samples : {len(fnames)}')
logging.info(f'Filenames obtained: {len(fnames_per_proc)}')
logging.info(f'Merging Atmosphere and Ocean 3d/2d tensors in one and save to disk')

c = 0
for fname in fnames_per_proc:
    # load data from filename
    atm_3d = np.load(os.path.join(atm_3d_src, fname)).reshape(-1, *input_shape)
    atm_2d = np.expand_dims(np.load(os.path.join(atm_2d_src, fname)), axis=1).reshape(-1, *input_shape)
    oce_3d = np.load(os.path.join(oce_3d_src, fname)).reshape(-1, *input_shape)
    oce_2d = np.expand_dims(np.load(os.path.join(oce_2d_src, fname)), axis=1).reshape(-1, *input_shape)
    # concat in a single tensor
    atm_oce_tensor = np.concatenate([atm_3d, atm_2d, oce_3d, oce_2d], axis=0)
    # convert the tensor to torch
    atm_oce_tensor = torch.as_tensor(atm_oce_tensor, dtype=torch.float32).unsqueeze(0)
    if target_resolution != '025':
        atm_oce_tensor = F.interpolate(
            atm_oce_tensor, 
            target_shape, 
            mode=interpolation_mode, 
            align_corners=True
        )
    atm_oce_tensor = atm_oce_tensor.squeeze(0).numpy()
    # save to disk
    with open(os.path.join(dst_dir, fname), 'wb') as f:
        np.save(f, atm_oce_tensor)
    # log
    logging.info(f'   [{c+1}/{len(fnames_per_proc)}] File {fname} successfully interpolated and stored on disk...')
    c += 1

logging.info(f'   All files processed')

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Tensors successfully merged. Process completed.')

# finalize the processes
MPI.Finalize()
    