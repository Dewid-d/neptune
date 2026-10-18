# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# ----------------------------------------------------------------------------

from mpi4py import MPI
import xarray as xr
import numpy as np
import argparse
import logging
import glob
import sys
import os

MAIN_DIR = 'PATH-TO-PROJECT-DIR'
sys.path.append(os.path.join(MAIN_DIR, 'library'))

# custom imports
from atmocean.utils import round_robin_split
from atmocean.macros import YEARS, ALL_DRIVERS

# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# parse the data domain
parser = argparse.ArgumentParser()
parser.add_argument('--domain', dest='domain', type=str, help='Specify the domain (atmosphere/ocean)')
args = parser.parse_args()
domain = args.domain

# get the years
years = YEARS

# get the domain
if domain == 'atmosphere':
    z = ALL_DRIVERS['levels']
elif 'ocean' in domain:
    z = ALL_DRIVERS['depths']
else:
    raise ValueError('Unsupported domain parameter. Available options: [atmosphere/ocean]')

fields_3d = ALL_DRIVERS['input'][domain]['fields_3d']
fields_2d = ALL_DRIVERS['input'][domain]['fields_2d']
src_dir = os.path.join(MAIN_DIR, f'data/datasets/{domain}/raw_data')
dst_3d_dir = os.path.join(MAIN_DIR, f'data/datasets/{domain}/processed/3D')
dst_2d_dir = os.path.join(MAIN_DIR, f'data/datasets/{domain}/processed/2D')
log_dir = os.path.join(MAIN_DIR, f'logs/processing/{domain}')

# only rank 0 will create the folders
if not rank:
    os.makedirs(dst_2d_dir, exist_ok=True)
    os.makedirs(dst_3d_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger 
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank+1}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# log
logging.info(f'Execution started')

# get the list of filenames in the src_dir
if domain == 'ocean':
    # ocean branch
    fnames = sum([glob.glob(os.path.join(src_dir, f'{year}/*/*.nc')) for year in years], [])
else:
    # atmosphere branch
    fnames = sum([glob.glob(os.path.join(src_dir, f'{year}*.nc')) for year in years], [])

# split the whole filenames list into "n_proc" sublists in a round robin fashion 
fnames_per_proc = round_robin_split(fnames, size)[rank]

# log
logging.info(f'Source directory: {src_dir}')
logging.info(f'Total filenames: {fnames}')
logging.info(f'Filenames assigned to this process: {fnames_per_proc}')

latitudes_padded = np.arange(-90, 90.25, 0.25)[::-1]

cnt = 0
for fname in fnames_per_proc:
    out_fname = os.path.basename(fname).replace('-', '').replace('.nc', '.npy')
    out_3d_fpath = os.path.join(dst_3d_dir, out_fname)
    out_2d_fpath = os.path.join(dst_2d_dir, out_fname)

    if (os.path.isfile(out_3d_fpath) and os.path.isfile(out_2d_fpath)): 
        # log
        logging.info(f'File {out_3d_fpath} and {out_2d_fpath} already existing. Skipping')
        continue

    try:
        if domain == 'ocean':
            # ocean branch
            ds = xr.load_dataset(fname, engine='h5netcdf')
            # select z levels
            ds = ds.sel(depth=z, method='nearest')
            # convert longitude from -180,180 to 0,360
            ds = ds.assign_coords(longitude=(ds.longitude + 360) % 360)
            # sort the longitude
            ds = ds.sortby('longitude')
            # assign a new coordinate for padded latitudes (on the Antartic region)
            ds = ds.assign_coords(coords={'latitude_pad': latitudes_padded})
            # create 3d and 2d arrays for padding
            pad_3d = np.full(shape=(1,len(z),40,1440), fill_value=np.nan)
            pad_2d = np.full(shape=(1,40,1440), fill_value=np.nan)
            
            # re-assign padded latitude to 3d fields
            for var in fields_3d:
                data = ds[var].data
                data = data[:,:,::-1]
                ds[var] = (('time', 'depth', 'latitude_pad', 'longitude'), np.concatenate([data, pad_3d], axis=2))
            
            # re-assign padded latitude to 2d fields
            for var in fields_2d:
                data = ds[var].data
                data = data[:,::-1]
                ds[var] = (('time', 'latitude_pad', 'longitude'), np.concatenate([data, pad_2d], axis=1))
            
            ds = ds.drop_dims(drop_dims='latitude')
            ds = ds.rename_dims(dims_dict={'latitude_pad':'latitude'})
            ds = ds.rename_vars({'latitude_pad':'latitude'})
            
            data_2d = ds[fields_2d].isel(time=0, missing_dims='ignore').to_array().data   
            data_3d = ds[fields_3d].isel(time=0, missing_dims='ignore').to_array().data
        else:
            # atmosphere branch
            ds = xr.load_dataset(fname)
            # select z levels
            ds = ds.sel(level=z) 
            
            data_2d = ds[fields_2d].isel(time=0, missing_dims='ignore').to_array().data   
            data_3d = ds[fields_3d].isel(time=0, missing_dims='ignore').to_array().data
            
        # save to npy 3d fields
        with open(out_3d_fpath, 'wb') as f3d:
            np.save(f3d, data_3d)

        # save to npy 2d fields
        with open(out_2d_fpath, 'wb') as f2d:
            np.save(f2d, data_2d)
        
        cnt += 1
    
    except Exception as e:
        logging.info(f'The following error occurred: {e}')
        continue
    
    # log
    logging.info(f'   [{cnt}/{len(fnames_per_proc)}]    File {os.path.basename(out_fname)} successfully converted')

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Conversion completed')

# finalize the processes
MPI.Finalize()