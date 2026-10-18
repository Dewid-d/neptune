# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Gabriele Accarino
# ----------------------------------------------------------------------------
from mpi4py import MPI
import numpy as np
import argparse
import calendar
import logging
import wget
import sys
import os

MAIN_DIR = 'PATH-TO-PROJECT-DIRECTORY'
sys.path.append(os.path.join(MAIN_DIR, 'library'))

# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# init folders
root_dir = os.path.join(MAIN_DIR, 'data/datasets/ocean')
data_dir = os.path.join(root_dir, 'raw_data')
log_dir = os.path.join(MAIN_DIR, 'logs/gathering/ocean')
base_url = "https://s3.waw3-1.cloudferro.com/mdl-native-14/native/GLOBAL_MULTIYEAR_PHY_ENS_001_031/cmems_mod_glo_phy-all_my_0.25deg_P1D-m_202311/"

# all the processes get the same year
parser = argparse.ArgumentParser()
parser.add_argument('--year', dest='year', type=str, help='Specify the year to download')
args = parser.parse_args()
year = args.year

# only rank 0 will create the folders
if not rank:
    logging.info(f'Processing {year}...')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger 
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank+1}-{year}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# log
logging.info(f'Execution started')

# each of the 12 processes get a different month
months = [f'{m:02d}' for m in np.arange(1, 13)]
month = months[rank]
days_in_month = calendar.monthrange(int(year), int(month))[1]

# organize downloaded data into {year}/{month} folders
os.makedirs(f'{data_dir}/{year}/{month}', exist_ok=True)

# request one month of data (daily) by each process 
for day in range(1, days_in_month+1):
    out_fname = f'{data_dir}/{year}/{month}/{year}{month}{day:02d}.nc'
    try:
        if os.path.isfile(out_fname):
            continue
        else:
            wget.download(url=base_url + f'{year}/{month}/cmems_mod_glo_phy-all_my_0.25deg_P1D-m-{year}{month}{day:02d}.nc', out=f'{data_dir}/{year}/{month}/{year}{month}{day:02d}.nc')

    except Exception as e:
        logging.info(f'The following error occurred: {e}')
        continue

    # log
    logging.info(f'   File {year}-{month}-{day:02d} successfully downloaded')
    
# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

if not rank:
    logging.info(f'{year} completed')
    
# finalize the processes
MPI.Finalize()