# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import argparse
import logging
import sys
import os

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Info + Parse CLI params
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

PROGRAM_NAME = r'''
    _   __           __                 
   / | / /__  ____  / /___  ______  ___ 
  /  |/ / _ \/ __ \/ __/ / / / __ \/ _ \
 / /|  /  __/ /_/ / /_/ /_/ / / / /  __/
/_/ |_/\___/ .___/\__/\__,_/_/ /_/\___/ 
          /_/                           
                                        
'''
PROGRAM_DESCRIPTION = 'Inference of Neptune on test set year'
PROGRAM_ARGUMENTS = [
    [('-m', '--model'), {'type':str, 'help':'Name of the Experiment', 'required':True}],
    [('-y', '--year'), {'type':int, 'help':'Year to inference', 'required':True}],
    [('-e', '--ensemble'), {'type':int, 'help':'Number of ensemble elements', 'required':True}],
    [('-o', '--ocean'), {'type':str, 'default':'default', 'choices':['default','oras5','cglors','glorys'], 'help':'Select the oceanic dataset to use', 'required':True}],
    [('-a', '--atmos'), {'type':str, 'default':'era5', 'choices':['era5'], 'help':'Select the atmospheric dataset to use', 'required':True}],
    [('-r', '--res'), {'type':str, 'default':'1', 'choices':['1','05','025'], 'help':'Select data resolution', 'required':True}],
]

parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
for arg in PROGRAM_ARGUMENTS:
    names, kwargs = arg
    sname, lname = names
    parser.add_argument(sname, lname, **kwargs)
args = parser.parse_args()


ocean = args.ocean
atmos = args.atmos
os.environ['ocean'] = ocean.upper()
os.environ['atmos'] = atmos.upper()

MAIN_DIR = 'PATH-TO-PROJECT-DIR'
sys.path.append(os.path.join(MAIN_DIR, 'library'))

from atmocean.dataset import NeptuneForecastDataset
from atmocean.forecast import *
from atmocean.models import *
import warnings
warnings.filterwarnings('ignore')


# get the forecasting horizon for this run
selected_model = args.model
config_src = os.path.join(MAIN_DIR, f'src/config/{selected_model}.toml')
forecast_horizon = 60
ensemble_size = args.ensemble
year = args.year
start_date = f'{year}-01-01'
end_date = f'{year}-12-31'
res = args.res

if res == '1':
    shape = (181, 360)
elif res == '05':
    shape = (361, 720)
elif res == '025':
    shape = (721, 1440)
tdim = 2
verbose = False

out_dir = os.path.join(MAIN_DIR, f'workflows/inference/img/{selected_model}')
inference_src = os.path.join(MAIN_DIR, f'data/forecasts/{selected_model}-from_{start_date}_to_{end_date}-ENS-{ensemble_size:02d}.zarr')
if ocean == 'default':
    sealand_mask_src = os.path.join(MAIN_DIR, f'data/masks/sealand_{res}_deg/sealand_mask_3d.npy')
    data_dir = os.path.join(MAIN_DIR, f'data/datasets/all_{res}_deg')
else:
    sealand_mask_src = os.path.join(MAIN_DIR, f'data/masks/sealand_{res}_deg/sealand_mask_3d-{ocean}.npy')
    data_dir = os.path.join(MAIN_DIR, f'data/datasets/all_{res}_deg-{ocean}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# log
print(f'Initializing the Model')

# initialize the ML model
model, config, state_dict = load_model_from_config(
    config_src=config_src, 
    main_dir=MAIN_DIR, 
    to_eval=True, 
    log_summary=True, 
    verbose=True, 
    device=device, 
)
model.ensemble_size = ensemble_size # we overwrite the ensemble size
insteps = config.train.insteps
outsteps = config.train.outsteps
lead_time = config.train.lead_time
drivers = config.data.drivers
fill_value = config.data.fill_value
epsilon = config.data.epsilon
torch.set_grad_enabled(False)
dates = pd.date_range(start_date, end_date, freq='1D')

# log
print(f'Initializing the Dataset')

# initilize the dataset
dataset = NeptuneForecastDataset(
    src = data_dir,
    drivers = drivers,
    start_date = start_date,
    end_date = end_date,
    insteps = insteps,
    outsteps = outsteps,
    lead_time = lead_time,
    forecast_horizon = forecast_horizon,
    fill_value = fill_value,
    scale = True,
    epsilon = epsilon,
)

# get the sealand mask
sealand_mask = np.load(sealand_mask_src)

print(f'Creating Empty Zarr Forecast Dataset')
ds = create_forecast_xr_dataset_from_torch_dataset(
    dataset = dataset,
    drivers = drivers,
    rollout = forecast_horizon,
    shape = shape,
    ensemble_size = ensemble_size,
)
print(f'Storing the Zarr Archive')
if not os.path.exists(inference_src):
    ds.isel(time=slice(0,1)).to_zarr(
        inference_src, mode="w", compute=True
    )
    last_written_date = None
else:
    ds = xr.open_zarr(inference_src)
    last_written_date = ds.time.values[-1]

print(f'Forecasting Dataset')
for i, date in enumerate(tqdm(dates)):
    if last_written_date is not None:
        if (pd.to_datetime(last_written_date) - pd.to_datetime(date)).days >= 0:
            # skip already written dates
            continue
    # forecast the time series and update the zarr
    (ic_a3d, ic_a2d, ic_o3d, ic_o2d), (fc_a3d, fc_a2d, fc_o3d, fc_o2d) = forecast_time_series(
        model = model, 
        dataset = dataset, 
        date = date, 
        insteps = insteps, 
        outsteps = outsteps, 
        lead_time = lead_time, 
        forecast_horizon = forecast_horizon, 
        epsilon = epsilon, 
        tdim = tdim, 
        verbose = verbose, 
        device = device, 
    )
    # create the forecast dataset
    ds = create_ds_with_forecasts(
        fcs = (fc_a3d, fc_a2d, fc_o3d, fc_o2d), 
        ics = (ic_a3d, ic_a2d, ic_o3d, ic_o2d), 
        drivers = drivers, 
        ic_date = date, 
        shape = shape, 
        rollout = forecast_horizon, 
        sealand_mask = sealand_mask, 
    )
    # update the zarr in the correct position
    ds.to_zarr(
        inference_src, 
        append_dim='time', 
    )

# log
print(f"Process completed")
