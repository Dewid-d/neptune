# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# ----------------------------------------------------------------------------

from pyperlin import FractalPerlin2D
from torch.utils.data import Dataset
from typing import List, Any
import pandas as pd
import numpy as np
import torch
import glob
import os

import warnings
warnings.filterwarnings('ignore')

from atmocean.utils import get_day_of_year, get_padded_shape, deaggregate_systems, aggregate_systems
from atmocean.macros import ALL_DATA_VARS, MEAN_DATA_VARS, STD_DATA_VARS, ATMOS_3D_VARS, ATMOS_2D_VARS, OCEAN_3D_VARS, OCEAN_2D_VARS
from atmocean.scaling import normalize



class NeptuneDataset(Dataset):
    def __init__(self, 
        src: str,                # source directory to draw the numpys
        years: List[int],        # list of dataset years
        drivers: dict,           # drivers dictionary contained in numpy
        insteps: int = 1,        # number of input steps
        outsteps: int = 1,       # number of output steps
        lead_time: int = 1,      # lead time between each step
        rollout: int = 1,        # number of rollout steps
        fill_value: float = 0.0, # fill value for nan
        scale: bool = False,     # whether to scale the data or not
        epsilon: float = 1e-7,   # epsilon value to correct small stds
        noise: float = False,    # whether to add perlin noise
        dtype: Any = None,       # pytorch data type 
        avg_atmos: bool = False, # whether to average atmosphere data
    ) -> None:
        super().__init__()
        self.src = src
        self.years = years
        self.drivers = drivers
        self.insteps = insteps
        self.outsteps = outsteps
        self.lead_time = lead_time
        self.rollout = rollout
        self.fill_value = fill_value
        self.scale = scale
        self.epsilon = epsilon
        self.noise = noise
        self.fractal_noise = None
        self.avg_atmos = avg_atmos
        self.dtype = dtype if dtype is not None else torch.float32
        # init all filenames lists
        self.fnames = []
        # get all the filenames for each year
        for year in years: 
            self.fnames += sorted(glob.glob(os.path.join(src, f'{year}*.npy')))
        # store cuda generator if noise is provided
        if noise:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.g_cuda = torch.Generator(device=device)
        # get the total length of the dataset
        self.n = len(self.fnames)
    
    def __len__(self):
        return self.n - self.insteps - self.outsteps * self.rollout * self.lead_time  - (self.lead_time - 1)
    
    def __getitem__(self, idx) -> Any:
        # shift index to take into account the time_steps
        index = idx + self.insteps * self.lead_time - 1
        # retrieve batch
        data, day, drivers, files = self._prepare_batch(index)
        return data, day, drivers, files
    
    def _prepare_batch(self, idx):
        # load and prepare input/output data from disk
        drivers = self.drivers
        files = self.fnames[idx - self.insteps * self.lead_time + 1 : idx + self.outsteps * self.rollout * self.lead_time + 1]
        if not self.avg_atmos:
            files = files[::self.lead_time]
            data = self._load_unpack(files)
            day = self._add_day_of_year(files)
            return data, day, drivers, files
        else:
            # we read much more data and now we have 
            # to compute a mean over atmosphere data
            # and get instantaneous data of the ocean
            data = self._load_unpack(files)
            day = self._add_day_of_year(files)
            a3d, a2d, o3d, o2d = deaggregate_systems(
                data.unsqueeze(0), # B x V x T x H x W
                ATMOS_3D_VARS, ATMOS_2D_VARS, OCEAN_3D_VARS, OCEAN_2D_VARS, 
            )
            # average a3d and a2d atmosphere
            a3d = torch.stack([torch.mean(xi, dim=2) for xi in torch.chunk(a3d, chunks=self.insteps + self.outsteps, dim=2)], dim=2) if a3d is not None else None
            a2d = torch.stack([torch.mean(xi, dim=2) for xi in torch.chunk(a2d, chunks=self.insteps + self.outsteps, dim=2)], dim=2) if a2d is not None else None
            o3d = torch.stack([xi[:,:,-1] for xi in torch.chunk(o3d, chunks=self.insteps + self.outsteps, dim=2)], dim=2) if o3d is not None else None
            o2d = torch.stack([xi[:,:,-1] for xi in torch.chunk(o2d, chunks=self.insteps + self.outsteps, dim=2)], dim=2) if o2d is not None else None
            data = aggregate_systems(a3d, a2d, o3d, o2d)
            data = data.squeeze(0)
            return data, day, drivers, files
        # return data, day, drivers, files
    
    def _add_day_of_year(self, files):
        doy = torch.cat([
            torch.as_tensor([get_day_of_year(file)]) 
            for file in files
        ])
        return doy
    
    def _load_unpack(self, files):
        tdim = 1
        # read all the timesteps and stack them
        data = torch.stack([torch.as_tensor(np.load(file, mmap_mode='c'), dtype=self.dtype) for file in files], dim=tdim)
        # get only the data belonging to the data vars
        batch = torch.stack([self._prepare_data(data[i], var) for i,var in enumerate(ALL_DATA_VARS)])
        return batch
    
    def _prepare_data(self, data, var):
        # scale the data with its own scaler
        if self.scale:
            data = (data - MEAN_DATA_VARS[var]) / (STD_DATA_VARS[var] + self.epsilon)
        # remove nan values
        data = torch.nan_to_num(data, nan=self.fill_value)
        # add perlin noise
        if self.noise:
            data = self._add_noise(data)
        return data
    
    def _add_noise(self, data: torch.Tensor):
        device = data.device # get the data device
        T, H, W = data.shape # get data shape
        sh = get_padded_shape((H, W), (32,32))
        resolutions = [(2**i,2**i) for i in range(1,7)]
        factors = [0.5**i for i in range(6)]
        if self.fractal_noise is None:
            self.fractal_noise = FractalPerlin2D(shape=(T, *sh), resolutions=resolutions, factors=factors, generator=self.g_cuda)
        noise =  self.fractal_noise
        noise_tensor = noise()[:,:H,:W].to(device) # create a noise tensor
        noise_tensor[self.insteps:] = 0.0 # zero the noise for the ground truth
        return data + noise_tensor # return the noisy data



class NeptuneForecastDataset(Dataset):
    def __init__(self, 
        src: str,                # source directory to draw the numpys
        drivers: dict,           # drivers dictionary contained in numpy
        start_date: str,         # start date of the forecast period
        end_date: str,           # end date of the forecast period
        insteps: int,            # number of input steps to the model
        outsteps: int,           # number of output steps to the model
        lead_time: int,          # lead time between each step
        forecast_horizon: int,   # number of forecasts for each date
        fill_value: float = 0.0, # fill value for nan
        scale: bool = False,     # whether to scale the data or not
        epsilon: float = 1e-7,   # epsilon value to correct small stds
        dtype: Any = None,       # pytorch data type 
    ) -> None:
        super().__init__()
        self.src = src
        self.drivers = drivers
        self.start_date = start_date
        self.end_date = end_date
        self.insteps = insteps
        self.outsteps = outsteps
        self.lead_time = lead_time
        self.forecast_horizon = forecast_horizon
        self.fill_value = fill_value
        self.scale = scale
        self.epsilon = epsilon
        self.dtype = dtype if dtype is not None else torch.float32
        start_date_offset = pd.to_datetime(start_date) - pd.DateOffset(days=(insteps * lead_time - 1))
        end_date_offset = pd.to_datetime(end_date) + pd.DateOffset(days=(forecast_horizon - lead_time))
        all_dates = pd.date_range(start_date_offset, end_date_offset, freq='1D')
        dates = pd.date_range(start_date, end_date, freq='1D')
        self.dates, self.all_dates = dates, all_dates
        self.fnames = [os.path.join(src, f'{date.year}{date.month:02d}{date.day:02d}.npy') for date in all_dates]
        assert forecast_horizon % lead_time == 0, f"forecast_horizon ({forecast_horizon}) is not a multiple of lead_time ({lead_time})"
    
    def __len__(self):
        return len(self.dates)
    
    # NOTE: deve restituire tutta la serie temporale di input che mi consente di produrre un forecast di `forecast_horizon` giorni
    def __getitem__(self, date) -> Any:
        # index = int(np.argwhere(self.all_dates == date)[0][0])
        # retrieve batch
        data, day, drivers, files, dates, idx = self._prepare_batch(date)
        return data, day, drivers, files, dates, idx
    
    def _prepare_batch(self, date):
        # load and prepare input/output data from disk
        drivers = self.drivers
        # convert date to datetime
        date = pd.to_datetime(date)
        start_date = date - pd.DateOffset(self.insteps, days=self.lead_time)
        end_date = date + pd.DateOffset(self.forecast_horizon, days=1)
        # get the dates needed for the time series
        dates = pd.date_range(start_date, end_date, freq='1D')
        # get the filenames to load
        files = [os.path.join(self.src, date.strftime('%Y%m%d') + '.npy') for date in dates]
        # prepare the data from disk
        data = self._load_unpack(files)
        day = self._add_day_of_year(files)
        index = self.insteps * self.lead_time - 1 # TODO: CAN REMOVE
        return data, day, drivers, files, dates, index
    
    def _add_day_of_year(self, files):
        doy = torch.cat([
            torch.as_tensor([get_day_of_year(file)]) 
            for file in files
        ])
        return doy
    
    def _load_unpack(self, files):
        tdim = 1
        # read all the timesteps and stack them
        data = torch.stack([torch.as_tensor(np.load(file, mmap_mode='r'), dtype=self.dtype) for file in files], dim=tdim)
        # get only the data belonging to the data vars
        batch = torch.stack([self._prepare_data(data[i], var) for i,var in enumerate(ALL_DATA_VARS)])
        return batch
    
    def _prepare_data(self, data, var):
        # scale the data with its own scaler
        if self.scale:
            data = normalize(data, var, self.epsilon)
        # remove nan values
        data = torch.nan_to_num(data, nan=self.fill_value)
        return data
