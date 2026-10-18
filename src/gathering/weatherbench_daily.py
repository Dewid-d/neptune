# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

from mpi4py import MPI
import xarray as xr
import pandas as pd
import logging
import sys
import os

MAIN_DIR = 'PATH-TO-PROJECT-DIRECTORY'
sys.path.append(os.path.join(MAIN_DIR, 'library'))
from atmocean.macros import YEARS
from atmocean.utils import is_xarray_netcdf_consistent

WEATHERBENCH2_ERA5_DAILY_025_ZARR = 'gs://weatherbench2/datasets/era5_daily/1959-2023_01_10-full_37-1h-0p25deg-chunk-1-s2s.zarr/'
WEATHERBENCH2_SINGLE_LEVEL_VARS = ['2m_temperature', 'surface_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', 'sea_surface_temperature', 'sea_ice_cover', 'mean_sea_level_pressure', 'total_column_water', 'total_precipitation_24hr']
WEATHERBENCH2_PRESSURE_LEVEL_VARS = ['geopotential', 'specific_humidity', 'potential_vorticity', 'vertical_velocity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
WEATHERBENCH2_PRESSURE_LEVELS = [300, 500, 600, 700, 850, 925, 1000]

# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# init folders
data_dir = os.path.join(MAIN_DIR, 'data/datasets/atmosphere/raw_data')
log_dir = os.path.join(MAIN_DIR, 'logs/gathering/atmosphere')

# select start and end year from the macros
start_year, end_year = YEARS[0], YEARS[-1]

# get start and end date
start_date = f'{start_year}-01-01'
end_date = f'{end_year}-12-31'

# create a list of dates
dates = pd.date_range(start_date, end_date, freq='1D')

# only rank 0 will create the folders
if not rank:
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger 
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# log
logging.info(f'Execution started')

# open weather-bench dataset
ds = xr.open_zarr(WEATHERBENCH2_ERA5_DAILY_025_ZARR)

# log
logging.info(f'Weatherbench2 dataset initialized')

# log
logging.info(f'We have a total of {ds.time.shape[0]} data maps to be collected from Weatherbench2')

# select the variables and the pressure level needed
ds = ds.sel(time=dates)[WEATHERBENCH2_SINGLE_LEVEL_VARS + WEATHERBENCH2_PRESSURE_LEVEL_VARS].sel(level=WEATHERBENCH2_PRESSURE_LEVELS)

# log
logging.info(f'Got subset of the Weatherbench2 dataset')

# loop over each date
for i, date in enumerate(dates):

    # if this condition is not valid, skip
    if i % size == rank:

        # create output filename
        path = os.path.join(data_dir, f'{date.year}-{date.month:02d}-{date.day:02d}.nc')

        if os.path.exists(path):
            # log
            logging.info(f'Dataset already exists at {path}. Testing consistency')

            # if file is consistent we can continue
            if is_xarray_netcdf_consistent(path=path):
                continue

        # select only the current iso time
        sel_ds = ds.sel(time=slice(date,date))

        # log
        logging.info(f'Downloading file for date {date} at {path}')

        # store the dataset to disk
        sel_ds.to_netcdf(path)

        # log
        logging.info(f'   File for date {date} successfully donwloaded')

        # check consistency of the downloaded sample
        is_xarray_netcdf_consistent(path)

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()
