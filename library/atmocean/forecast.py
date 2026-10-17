# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

from atmocean.utils import deaggregate_systems, realign_drivers_dict
from atmocean.scaling import unnormalize
from atmocean.macros import ALL_DRIVERS
# from atmocean.losses import *

from torch.utils.data import Dataset
from einops import rearrange
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import xarray as xr
import numpy as np
import torch
import os


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def move_data_to_device(x: tuple, device='cpu'):
    return tuple([xi.to(device) if xi is not None else None for xi in x])


# # NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/rollout_forecast.ipynb)
# def roll_tensors(x, shifts, dims):
#     return tuple([torch.roll(xi, shifts=shifts, dims=dims) if xi is not None else None for xi in x])


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/rollout_forecast.ipynb)
def shift_tensors(x: tuple[torch.Tensor], shift: int, dim: int):
    """
    Shifts the input tensors in order to accomodate the output inside the input and rollout the predictions
    """
    out = []
    for xi in x:
        if xi is None:
            out.append(None)
        else:
            shifted = xi.narrow(dim, shift, xi.size(dim) - shift) # .detach()
            pad_shape = list(shifted.shape)
            pad_shape[dim] = shift
            pad = torch.zeros(pad_shape, dtype=xi.dtype, device=xi.device)
            out.append(torch.cat([shifted, pad], dim=dim))
    return tuple(out)


def update_input(x, y, outsteps):
    xo = []
    for xi, yi in zip(x, y):
        if xi is not None: xi[:, :, -outsteps:] = yi
        xo.append(xi)
    return xo


def update_forcing(x, X, i, insteps):
    return [Xi[:, :, i-insteps+1 : i+1] if xi is None else xi for xi, Xi in zip(x, X)]

def update_input_with_forcing(
        x:        list[torch.Tensor],
        y:        list[torch.Tensor],
        X:        list[torch.Tensor],
        i:        int,
        insteps:  int,
        outsteps: int,
        drivers:  dict,
    ):
    drvs_list = (
        (drivers['input']['atmos']['fields_3d'], drivers['output']['atmos']['fields_3d']),
        (drivers['input']['atmos']['fields_2d'], drivers['output']['atmos']['fields_2d']),
        (drivers['input']['ocean']['fields_3d'], drivers['output']['ocean']['fields_3d']),
        (drivers['input']['ocean']['fields_2d'], drivers['output']['ocean']['fields_2d']),
    )
    xo = []
    for s, (xi, yi, Xi) in enumerate(zip(x, y, X)):
        # prediction none, ground truth none
        if yi is None and Xi is None:
            xi_new = None
        # prediction none, ground truth ok
        elif yi is None and Xi is not None:
            xi_new = select_input([Xi, ], i, insteps)[0]
        # prediction ok, ground truth ok
        elif yi is not None and Xi is not None:
            # avoid inplace operation that could break computational graph
            xi_new = xi.clone()
            # number of variables is the same between input and output
            if xi.shape[1] == yi.shape[1]:
                # TRUNCATED BPTT: la predizione entra come input del passo
                # successivo ma senza portarsi dietro il grafo computazionale
                xi_new[:, :, -outsteps:] = yi.detach()
            # num variables is different between input and output
            else:
                in_drvs, out_drvs = drvs_list[s]
                for j, field in enumerate(in_drvs):
                    if field in out_drvs:
                        k = np.argwhere(np.array(out_drvs) == field)[0][0]
                        xi_new[:, j, -outsteps:] = yi[:, k].detach()
                    else:
                        xi_new[:, j] = select_input([Xi, ], i, insteps)[0][:, j]
        xo.append(xi_new)
    return xo

# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/rollout_forecast.ipynb)
def select_input(x, i, insteps):
    return [xi[:, :, i-insteps+1 : i+1] if xi is not None else None for xi in x]


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/rollout_forecast.ipynb)
def select_output(x, i, outsteps):
    return [xi[:, :, i+1 : i+1+outsteps] if xi is not None else None for xi in x]


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def stack_forecasts(forecasts, dim: int = 2):
    stacked_forecasts = torch.cat(forecasts, dim=dim) if forecasts[0] is not None else None 
    if stacked_forecasts is not None:
        # prediction time-delta comes first as dimension
        stacked_forecasts = rearrange(stacked_forecasts, 'b v p z h w -> b p v z h w')
    return stacked_forecasts


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/rollout_forecast.ipynb)
def mean_prediction(x, is_ens: bool = False):
    xo = []
    for xi in x:
        if xi is not None:
            if is_ens:
                xo.append(torch.mean(xi, dim = -1))
            else:
                xo.append(xi)
        else:
            xo.append(None)
    return xo


def cum_sum_data(
        x: list[torch.Tensor],
        dy: list[torch.Tensor],
        outsteps: int,
        tdim: int = 2,
        is_ens: bool = False,
    ):
    yp = []
    for xi, dyi in zip(x, dy):
        if dyi is not None:
            # steps list
            yis = []
            # running output
            if not is_ens:
                yir = xi[:,:,-1:].detach()
            else:
                ens_size = dyi.shape[-1]
                yir = (
                    xi[:,:,-1:]
                    .unsqueeze(-1)
                    .expand(*xi[:,:,-1:].shape, ens_size)
                    .detach()
                )
            # for each outstep
            for k in range(outsteps):
                # sum for each outstep
                yir = yir + dyi[:,:,k:k+1]
                yis.append(yir)
                # block gradient between steps
                yir = yir.detach()
            ypi = torch.cat(yis, dim=tdim)
            yp.append(ypi)
        else:
            yp.append(None)
    return yp


def rescale(x, drivers: dict, epsilon: float):
    a3d, a2d, o3d, o2d = x # T x F x V x Z x H x W
    af3d, af2d, of3d, of2d = (drivers['output']['atmos']['fields_3d'], 
                              drivers['output']['atmos']['fields_2d'], 
                              drivers['output']['ocean']['fields_3d'], 
                              drivers['output']['ocean']['fields_2d'])
    for i,field in enumerate(af3d):
        if a3d is not None:
            for l,level in enumerate(drivers['levels']):
                a3d[:,:,i,l] = unnormalize(a3d[:,:,i,l], f'{field}_{level}', epsilon)
    for i,field in enumerate(af2d):
        if a2d is not None:
            a2d[:,:,i] = unnormalize(a2d[:,:,i], field, epsilon)
    for i,field in enumerate(of3d):
            if o3d is not None:
                for d,depth in enumerate(drivers['depths']):
                    o3d[:,:,i,d] = unnormalize(o3d[:,:,i,d], f'{field}_{int(depth)}', epsilon)
    for i,field in enumerate(of2d):
        if o2d is not None:
            o2d[:,:,i] = unnormalize(o2d[:,:,i], field, epsilon)
    return (a3d, a2d, o3d, o2d)


def permute_ics(ics):
    return [ic.permute(0,2,1,3,4,5) if ic is not None else None for ic in ics]


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def select_forecast_input(x, indices):
    try:
        return [xi[:, :, indices] if xi is not None else None for xi in x]
    except:
        for xi in x:
            if xi is not None:
                print(xi.shape, indices)


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def update_forecast_input_with_forcing(x, y, X, outsteps, indices, drivers):
    drvs_list = (
        (drivers['input']['atmos']['fields_3d'], drivers['output']['atmos']['fields_3d']),
        (drivers['input']['atmos']['fields_2d'], drivers['output']['atmos']['fields_2d']),
        (drivers['input']['ocean']['fields_3d'], drivers['output']['ocean']['fields_3d']),
        (drivers['input']['ocean']['fields_2d'], drivers['output']['ocean']['fields_2d']), 
    )
    xo = []
    for s,(xi, yi, Xi) in enumerate(zip(x, y, X)):
        # prediction none, ground truth none
        if yi is None and Xi is None:
            xi = None
        # prediction none, ground truth ok
        elif yi is None and Xi is not None:
            xi = Xi[:, :, indices]
        # prediction ok, ground truth ok
        elif yi is not None and Xi is not None:
            # number of variables is the same between input and output
            if xi.shape[1] == yi.shape[1]:
                xi[:, :, -outsteps:] = yi
            # num variables is different between input and output
            else:
                # get the in and out drivers for that system
                in_drvs, out_drvs = drvs_list[s]
                # iterate over each field in the system
                for j, field in enumerate(in_drvs):
                    # if the system var IS in output drivers
                    if field in out_drvs:
                        # get the index of that output driver
                        k = np.argwhere(np.array(out_drvs)==field)[0][0]
                        # update the input with that driver
                        xi[:, j, -outsteps:] = yi[:, k]
                    # if the system var IS NOT in the output drivers
                    else:
                        # update the input with ground truth drivers
                        xi[:, j] = Xi[:, j, indices]
        xo.append(xi)
    return xo


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def select_forecast_ics(x, index):
    return [xi[:,:,index:index+1] if xi is not None else None for xi in x]


def create_forecast_xr_dataset_from_torch_dataset(
    dataset: Dataset, 
    drivers: dict, 
    rollout: int, 
    shape: tuple = (181, 360), 
    ensemble_size: int = None, 
) -> xr.Dataset:
    data_shape = shape
    # create the coordinates dictionary
    coords = dict()
    if ensemble_size is not None: coords['ensemble_member'] = np.arange(ensemble_size)
    coords['time'] = dataset.dates[:1]
    coords['prediction_timedelta'] = pd.timedelta_range('0 day', periods=rollout+1)
    coords['latitude'] = np.linspace(-90.0, 90.0, shape[0])[::-1]
    coords['longitude'] = np.linspace(0, 359, shape[1])
    coords['level'] = drivers['levels']
    coords['depth'] = drivers['depths']
    # create the dataset using the coordinates
    ds = xr.Dataset(coords=coords)
    E, T, F, Z, D, H, W = (
        ds.ensemble_member.shape[0] if ensemble_size is not None else None, 
        ds.time.shape[0], 
        ds.prediction_timedelta.shape[0], 
        ds.level.shape[0], 
        ds.depth.shape[0], 
        ds.latitude.shape[0], 
        ds.longitude.shape[0]
    )
    if ensemble_size is None:
        dims = (('time', 'prediction_timedelta', 'level', 'latitude', 'longitude'), ('time', 'prediction_timedelta', 'latitude', 'longitude'), 
                ('time', 'prediction_timedelta', 'depth', 'latitude', 'longitude'), ('time', 'prediction_timedelta', 'latitude', 'longitude'))
        shapes = ((T, F, Z, H, W), (T, F, H, W), 
                  (T, F, D, H, W), (T, F, H, W))
    else:
        dims = (('ensemble_member', 'time', 'prediction_timedelta', 'level', 'latitude', 'longitude'), ('ensemble_member', 'time', 'prediction_timedelta', 'latitude', 'longitude'), 
                ('ensemble_member', 'time', 'prediction_timedelta', 'depth', 'latitude', 'longitude'), ('ensemble_member', 'time', 'prediction_timedelta', 'latitude', 'longitude'))
        shapes = ((E, T, F, Z, H, W), (E, T, F, H, W), 
                  (E, T, F, D, H, W), (E, T, F, H, W))
    fields = (drivers['output']['atmos']['fields_3d'], drivers['output']['atmos']['fields_2d'], 
              drivers['output']['ocean']['fields_3d'], drivers['output']['ocean']['fields_2d'])
    for field, shape, dim in zip(fields, shapes, dims):
        for f in field:
            ds[f] = (dim, np.empty(shape=shape))
    # define chunking size
    chunks = dict()
    if ensemble_size is not None: chunks['ensemble_member'] = 1
    chunks['time'] = 1
    chunks['prediction_timedelta'] = rollout + 1
    chunks['depth'] = len(drivers['depths'])
    chunks['level'] = len(drivers['levels'])
    chunks['latitude'] = data_shape[0]
    chunks['longitude'] = data_shape[1]
    # correctly chunk the dataset
    ds = ds.chunk(chunks=chunks)
    return ds

# NOTE: this version forces float32 datasets instead of float64 
def update_ds_with_forecasts(ds, fcs, ics, drivers, ic_date, sealand_mask=None):
    for fc in fcs:
        if fc is not None:
            E = fc.shape[0]
            break
    fc_a3d, fc_a2d, fc_o3d, fc_o2d = fcs
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = ics
    if fc_a3d is not None:
        for f, field in enumerate(drivers['output']['atmos']['fields_3d']):
            time_series = torch.cat((ic_a3d[:, :, f], fc_a3d[:, :, f]), dim=1).float()
            ds[field].loc[{'time': ic_date}] = time_series.numpy()
    if fc_a2d is not None:
        for f, field in enumerate(drivers['output']['atmos']['fields_2d']):
            time_series = torch.cat((ic_a2d[:, :, f, 0], fc_a2d[:, :, f, 0]), dim=1).float()
            ds[field].loc[{'time': ic_date}] = time_series.numpy()
    if fc_o3d is not None:
        depths = drivers['depths']
        all_depths = ALL_DRIVERS['depths']
        depth_ids = torch.as_tensor([i for i, d in enumerate(all_depths) if int(d) in [int(dpt) for dpt in depths]])
        sealand_mask = torch.as_tensor(sealand_mask).index_select(dim=0, index=depth_ids).numpy()
        for f, field in enumerate(drivers['output']['ocean']['fields_3d']):
            time_series = torch.cat((ic_o3d[:, :, f], fc_o3d[:, :, f]), dim=1).float().numpy()
            sealand_mask_3d = sealand_mask[np.newaxis, np.newaxis]
            time_series = np.where(sealand_mask_3d == 1, time_series, np.nan).astype(np.float32)
            ds[field].loc[{'time': ic_date}] = time_series
    if fc_o2d is not None:
        for f, field in enumerate(drivers['output']['ocean']['fields_2d']):
            time_series = torch.cat((ic_o2d[:, :, f, 0], fc_o2d[:, :, f, 0]), dim=1).float().numpy()
            sealand_mask_3d = sealand_mask[0][np.newaxis, np.newaxis]
            time_series = np.where(sealand_mask_3d == 1, time_series, np.nan).astype(np.float32)
            ds[field].loc[{'time': ic_date}] = time_series
    return ds

def create_ds_from_date(date, drivers, rollout, shape, ensemble_size):
    data_shape = shape
    coords = dict()
    # if ensemble_size > 1: coords['ensemble_member'] = np.arange(ensemble_size)
    coords['ensemble_member'] = np.arange(ensemble_size)
    coords['time'] = [date]
    coords['prediction_timedelta'] = pd.timedelta_range('0 day', periods=rollout+1)
    coords['latitude'] = np.linspace(-90.0, 90.0, shape[0])[::-1]
    coords['longitude'] = np.linspace(0, 359, shape[1])
    coords['level'] = drivers['levels']
    coords['depth'] = drivers['depths']
    ds = xr.Dataset(coords=coords)
    E, T, F, Z, D, H, W = (
        # ds.ensemble_member.shape[0] if ensemble_size > 1 else None, 
        ds.ensemble_member.shape[0],
        ds.time.shape[0], 
        ds.prediction_timedelta.shape[0], 
        ds.level.shape[0], 
        ds.depth.shape[0], 
        ds.latitude.shape[0], 
        ds.longitude.shape[0]
    )
    # if ensemble_size == 1:
    #     dims = (('time', 'prediction_timedelta', 'level', 'latitude', 'longitude'), ('time', 'prediction_timedelta', 'latitude', 'longitude'), 
    #             ('time', 'prediction_timedelta', 'depth', 'latitude', 'longitude'), ('time', 'prediction_timedelta', 'latitude', 'longitude'))
    #     shapes = ((T, F, Z, H, W), (T, F, H, W), 
    #               (T, F, D, H, W), (T, F, H, W))
    # else:
    dims = (('ensemble_member', 'time', 'prediction_timedelta', 'level', 'latitude', 'longitude'), ('ensemble_member', 'time', 'prediction_timedelta', 'latitude', 'longitude'), 
            ('ensemble_member', 'time', 'prediction_timedelta', 'depth', 'latitude', 'longitude'), ('ensemble_member', 'time', 'prediction_timedelta', 'latitude', 'longitude'))
    shapes = ((E, T, F, Z, H, W), (E, T, F, H, W), 
                (E, T, F, D, H, W), (E, T, F, H, W))
    fields = (drivers['output']['atmos']['fields_3d'], drivers['output']['atmos']['fields_2d'], 
              drivers['output']['ocean']['fields_3d'], drivers['output']['ocean']['fields_2d'])
    for field, shape, dim in zip(fields, shapes, dims):
        for f in field:
            ds[f] = (dim, np.empty(shape=shape, dtype=np.float32))
    # define the chunk sizes
    chunks = dict()
    # if ensemble_size > 1: 
    chunks['ensemble_member'] = 1
    chunks['time'] = 1
    chunks['prediction_timedelta'] = rollout + 1
    chunks['depth'] = len(drivers['depths'])
    chunks['level'] = len(drivers['levels'])
    chunks['latitude'] = data_shape[0]
    chunks['longitude'] = data_shape[1]
    # chunk the dataset with the correct chunking
    ds = ds.chunk(chunks=chunks)
    return ds


def create_ds_with_forecasts(fcs, ics, drivers, ic_date, shape, rollout, sealand_mask = None):
    for fc in fcs:
        if fc is not None:
            ensemble_size = fc.shape[0]
            break
    # create empty dataset to be filled with forecasts
    ds = create_ds_from_date(ic_date, drivers, rollout, shape, ensemble_size)
    # update the dataset with the forecasts
    ds = update_ds_with_forecasts(ds, fcs, ics, drivers, ic_date, sealand_mask)
    return ds


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: test/forecast_dataset.ipynb)
def forecast_dataset(
        dates: list,                # dates of the forecasts
        dataset: Dataset,           # dataset to be forecasted
        model: nn.Module,           # model to use
        ds: xr.Dataset,             # output forecast xarray dataset
        drivers: dict,              # drivers used during training
        insteps: int,               # number of input steps to the model
        outsteps: int,              # number of output steps to the model
        lead_time: int,             # lead time between each step
        forecast_horizon: int,      # number of rollouts (in days) to be produced
        tdim: int = 2,              # time dimension
        epsilon: float = 1e-7,      # epsilon to correct std
        device: str = 'cpu',        # device to use during inference
        verbose: bool = False,      # whether to print each step or not
        n_samples: int = None,      # number of samples to be forecasted
        sealand_mask = None,        # sealand mask to remove the land on ocean variables
    ):
    l = 0
    for date in tqdm(dates):
        if n_samples is not None and n_samples == l:
            break
        l += 1
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
        # put the forecasts to the correct place of the prediction dataset
        ds = update_ds_with_forecasts(ds, (fc_a3d, fc_a2d, fc_o3d, fc_o2d), (ic_a3d, ic_a2d, ic_o3d, ic_o2d), drivers, date, sealand_mask)
    return ds


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def create_data_dict(data, dates):
    """
    Creates a data dictionary containing, for each passed date, the 
    data belonging to that day, according to the key:values:
    >> date : (atmos_3d, atmos_2d, ocean_3d, ocean_2d)
    """
    # separate the systems
    atmos_3d, atmos_2d, ocean_3d, ocean_2d = data
    # create a data dictionary containing {date : (atmos_3d, atmos_2d, ocean_3d, ocean_2d)} items
    data_dict = {date : (
        atmos_3d[:,:,i] if atmos_3d is not None else None, 
        atmos_2d[:,:,i] if atmos_2d is not None else None, 
        ocean_3d[:,:,i] if ocean_3d is not None else None, 
        ocean_2d[:,:,i] if ocean_2d is not None else None, 
    ) for i, date in enumerate(dates)}
    return data_dict


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def get_day_of_year(date: str):
    '''Day and Month of the Year (DMoY)'''
    date = pd.to_datetime(date)
    day_of_year = pd.Period(date, freq='d').day_of_year
    return day_of_year


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def update_forecast_dict(dates, forecasts, forecast_dict, tdim: int = 2):
    for i,date in enumerate(dates):
        index = torch.as_tensor(i)
        forecast_dict[date] = [
            torch.index_select(forecast, tdim, index).squeeze(tdim) if forecast is not None else None
            for forecast in forecasts
            ]
    return forecast_dict


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def stack_data_from_dict(data_dict, dim: int = 0):
    """
    Stack of all elements of each position in the tuple within the data dictionary.
    The data dictionary is structured as follows:
    >> data_dict = {
    >>     '2020-01-01': (atmos_3d, atmos_2d, ocean_3d, ocean_2d), 
    >>     '2020-01-02': (atmos_3d, atmos_2d, ocean_3d, ocean_2d), 
    >>     ...
    >> }
    This function stacks each system together within the dictionary, 
    returning a tuple of Tensors like this:
    >> (atmos_3d_stacked, atmos_2d_stacked, ocean_3d_stacked, ocean_2d_stacked)
    Note: this code works also when some system is not provided (i.e., None)
    """
    values = list(data_dict.values())
    stacked = []
    for i in range(len(values[0])):
        elems = [v[i] for v in values if v[i] is not None]
        if elems:
            stacked.append(torch.stack(elems, dim=dim))
        else:
            stacked.append(None)
    return tuple(stacked)


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def load_input_data(dates, data_dict: dict, forecast_dict: dict, drivers: dict, verbose: bool = False):
    drvs_list = (
        (drivers['input']['atmos']['fields_3d'], drivers['output']['atmos']['fields_3d']),
        (drivers['input']['atmos']['fields_2d'], drivers['output']['atmos']['fields_2d']),
        (drivers['input']['ocean']['fields_3d'], drivers['output']['ocean']['fields_3d']),
        (drivers['input']['ocean']['fields_2d'], drivers['output']['ocean']['fields_2d']), 
    )
    data = {}
    for date in dates:
        if forecast_dict.get(date) is not None:
            fdata = forecast_dict[date]
            ddata = data_dict[date]
            xo = []
            for s, (fd, dd) in enumerate(zip(fdata, ddata)):
                # prediction none, ground truth none
                if fd is None and dd is None:
                    xi = None
                # prediction none, ground truth ok
                elif fd is None and dd is not None:
                    xi = dd
                # prediction ok, ground truth ok
                elif fd is not None and dd is not None:
                    # number of variables is the same between input and output
                    if fd.shape == dd.shape:
                        xi = fd
                    # number of variables is different between input and output
                    else:
                        # output is a clone of the ground truth
                        xi = torch.clone(dd)
                        # get the in and out drivers for that system
                        in_drvs, out_drvs = drvs_list[s]
                        # iterate over each field in the system
                        for j, field in enumerate(in_drvs):
                            # if the system var IS in output drivers
                            if field in out_drvs:
                                # get the index of that output driver
                                k = np.argwhere(np.array(out_drvs)==field)[0][0]
                                # update the input with that driver
                                xi[:, j] = fd[:, k]
                        print(xi.shape, torch.nansum(xi - dd))
                # add the data to x
                xo.append(xi)
            data[date] = tuple(xo)
            if verbose: print(f'   using `forecast_dict` for date {date}')
        else:
            data[date] = data_dict[date]
            if verbose: print(f'   using `data_dict` for date {date}')
    return data


# NOTE: (TESTED) - properly works (date: 2025-05-14, src: validation/reanalysis/inference_testing.ipynb)
def forecast_time_series(
        model, 
        dataset, 
        date, 
        insteps: int, 
        outsteps: int, 
        lead_time: int, 
        forecast_horizon: int, 
        epsilon: float = 1e-7, 
        tdim: int = 2, 
        verbose: bool = False, 
        use_tqdm: bool = False, 
        device = 'cpu', 
    ):
    # get the data from the dataset
    batch, _, drivers, files, dates_, _ = dataset[date]
    # get the model ensemble size
    E = model.ensemble_size
    # add B dimension
    batch:torch.Tensor = batch.unsqueeze(0)
    # broadcast the batch to the ensemble size
    batch = batch.broadcast_to((E, *batch.shape[1:]))
    # parse date
    date = pd.to_datetime(date)
    # deaggregate variables
    atmos_3d, atmos_2d, ocean_3d, ocean_2d = deaggregate_systems(
        batch, 
        drivers['input']['atmos']['fields_3d'], 
        drivers['input']['atmos']['fields_2d'], 
        drivers['input']['ocean']['fields_3d'], 
        drivers['input']['ocean']['fields_2d'], 
    )
    # get only the depth levels that we actually use
    depths, levels = drivers['depths'], drivers['levels']
    all_depths, all_levels = ALL_DRIVERS['depths'], ALL_DRIVERS['levels']
    depth_ids = torch.as_tensor([i for i,d in enumerate(all_depths) if int(d) in [int(dpt) for dpt in depths]])
    level_ids = torch.as_tensor([i for i,l in enumerate(all_levels) if int(l) in [int(lvl) for lvl in levels]])
    atmos_3d = atmos_3d.index_select(dim=tdim+1, index=level_ids) if atmos_3d is not None else None
    ocean_3d = ocean_3d.index_select(dim=tdim+1, index=depth_ids) if ocean_3d is not None else None
    # get the input dates
    input_dates = [pd.to_datetime(file.split('/')[-1].split('.')[0]).strftime('%Y-%m-%d') for file in files]
    # create a data dictionary containing {date : data} items
    data_dict = create_data_dict((atmos_3d, atmos_2d, ocean_3d, ocean_2d), input_dates)
    # forecast dates contains the Initial Condition + all the forecasting dates
    forecast_dates = [date.strftime('%Y-%m-%d') for date in pd.date_range(date, date + pd.DateOffset(forecast_horizon), freq='1D')]
    # create a data dictionary of forecasts containing {date: None} items
    forecast_dict = {date: None for date in forecast_dates[1:]}
    ic_date = date
    if verbose: print(f'Forecasting the Time Series that starts from {ic_date}')

    # create noise tensors to inject into the networks
    if model.is_ens and model.__class__.__name__ in ["AtlasOceaNOv2"]:
        noise = []
        D = model.dim
        B, _, _, H, W = batch.shape
        if model.conv_module == "spherical":
            nshape = (B * insteps, D, H, W)
        elif model.conv_module == "spectral":
            nshape = (B, D, insteps, H, W)
        noise = torch.randn(nshape, device=device, dtype=batch.dtype)
    else:
        noise = None

    if use_tqdm:
        iterator = tqdm(enumerate(forecast_dates[1:]), total=len(forecast_dates[1:]))
    else:
        iterator = enumerate(forecast_dates[1:])
    for i, date in iterator:
        # get the input dates to be used
        indates = [(pd.to_datetime(date) - pd.DateOffset(i+1, days=lead_time)).strftime('%Y-%m-%d') for i in reversed(range(insteps))]
        # get the output dates 
        outdates = [(pd.to_datetime(date) + pd.DateOffset(i, days=lead_time)).strftime('%Y-%m-%d') for i in range(outsteps)]
        if verbose: print(f'   predicting date {date} ({outdates})')
        # load the input data from the `forecast_dict` or the `data_dict`
        indata = load_input_data(indates, data_dict, forecast_dict, drivers, verbose)
        # stack the indata into a torch Tensor
        xa3d, xa2d, xo3d, xo2d = stack_data_from_dict(indata, dim=tdim)
        # get static time variable
        t = torch.cat([torch.as_tensor([get_day_of_year(date)]) for date in indates])
        t = t.unsqueeze(0)
        # broadcast the timestep to the ensemble size
        t = t.broadcast_to((E, *t.shape[1:]))
        # move to device
        xa3d, xa2d, xo3d, xo2d, t = move_data_to_device((xa3d, xa2d, xo3d, xo2d, t), device)
        if verbose: print(f'   data moved to device')
        # forward pass
        if verbose: print(f'   forecasting')
        ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = model(
            x=(xa3d, xa2d, xo3d, xo2d), 
            t=t, r=i, noise=noise, 
            drivers=drivers, 
            out_attn=False, 
            agg_systems=False
        )
        # remove last dim if 1
        # if ya3d_pred is not None: ya3d_pred = ya3d_pred[...,0] if ya3d_pred.shape[-1] == 1 else ya3d_pred
        # if ya2d_pred is not None: ya2d_pred = ya2d_pred[...,0] if ya2d_pred.shape[-1] == 1 else ya2d_pred
        # if yo3d_pred is not None: yo3d_pred = yo3d_pred[...,0] if yo3d_pred.shape[-1] == 1 else yo3d_pred
        # if yo2d_pred is not None: yo2d_pred = yo2d_pred[...,0] if yo2d_pred.shape[-1] == 1 else yo2d_pred
        # if the model is in residual configuration
        if model.is_residual:
            ya3d_pred = xa3d + ya3d_pred if ya3d_pred is not None else None
            ya2d_pred = xa2d + ya2d_pred if ya2d_pred is not None else None
            yo3d_pred = xo3d + yo3d_pred if yo3d_pred is not None else None
            yo2d_pred = xo2d + yo2d_pred if yo2d_pred is not None else None
        # offload to cpu
        ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = move_data_to_device(
            (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
            'cpu')
        xa3d, xa2d, xo3d, xo2d, t = move_data_to_device(
            (xa3d, xa2d, xo3d, xo2d, t), 
            'cpu')
        if verbose: print(f'   data offloaded to cpu')
        # update the forecasting dictionary
        forecast_dict = update_forecast_dict(
            outdates, 
            (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
            forecast_dict, 
            tdim
        )
    # clean the forecast dictionary from additional keys
    to_remove_keys = []
    for key in forecast_dict.keys():
        if key not in forecast_dates:
            to_remove_keys.append(key)
    # for each key to remove
    for key in to_remove_keys:
        # remove the key from the dictionary
        del forecast_dict[key]
    forecast_dict.keys()
    # get the initial conditions
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = data_dict[ic_date.strftime('%Y-%m-%d')]
    # add time dimension to initial conditions
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = (
        ic_a3d.unsqueeze(tdim) if ic_a3d is not None else None, 
        ic_a2d.unsqueeze(tdim) if ic_a2d is not None else None, 
        ic_o3d.unsqueeze(tdim) if ic_o3d is not None else None, 
        ic_o2d.unsqueeze(tdim) if ic_o2d is not None else None, 
    )
    # stack forecasts together
    (fc_a3d, fc_a2d, fc_o3d, fc_o2d) = stack_data_from_dict(forecast_dict, dim=tdim)
    if verbose: print(f'  Forecasts stacked')

    # (torch.Size([1, 4, 1, 14, 181, 360]), torch.Size([1, 4, 1, 1, 181, 360]))
    # permute data to match the xarray
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = (
        rearrange(ic_a3d, 't v f z h w -> t f v z h w') if ic_a3d is not None else None, 
        rearrange(ic_a2d, 't v f z h w -> t f v z h w') if ic_a2d is not None else None, 
        rearrange(ic_o3d, 't v f z h w -> t f v z h w') if ic_o3d is not None else None, 
        rearrange(ic_o2d, 't v f z h w -> t f v z h w') if ic_o2d is not None else None, 
    )
    fc_a3d, fc_a2d, fc_o3d, fc_o2d = (
        rearrange(fc_a3d, 't v f z h w -> t f v z h w') if fc_a3d is not None else None, 
        rearrange(fc_a2d, 't v f z h w -> t f v z h w') if fc_a2d is not None else None, 
        rearrange(fc_o3d, 't v f z h w -> t f v z h w') if fc_o3d is not None else None, 
        rearrange(fc_o2d, 't v f z h w -> t f v z h w') if fc_o2d is not None else None, 
    )

    # rescale to original feature scale
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = rescale((ic_a3d, ic_a2d, ic_o3d, ic_o2d), drivers, epsilon)
    fc_a3d, fc_a2d, fc_o3d, fc_o2d = rescale((fc_a3d, fc_a2d, fc_o3d, fc_o2d), drivers, epsilon)
    if verbose: print(f'  Forecasts rescaled to original feature scale')

    # return the list of forecasts of this time series
    return (ic_a3d, ic_a2d, ic_o3d, ic_o2d), (fc_a3d, fc_a2d, fc_o3d, fc_o2d)


# it is the same as forecast_time_series() but with additional parameters
def forecast_time_series_ablation_study(
        model, 
        dataset, 
        date, 
        insteps, 
        outsteps, 
        lead_time, 
        forecast_horizon, 
        atmospheric_forcing: str = 'default', 
        time_embedding: str = 'default', 
        dynamic_ocean: str = 'default', 
        epsilon: float = 1e-7, 
        tdim: int = 2, 
        verbose: bool = False, 
        use_tqdm: bool = False, 
        device = 'cpu', 
):
    # get the data from the dataset
    batch, _, drivers, files, dates_, _ = dataset[date]
    # add B dimension
    batch = batch.unsqueeze(0)
    # parse date
    date = pd.to_datetime(date)
    
    # atmospheric forcing processing
    if atmospheric_forcing == 'none':
        drivers['input']['atmos']['fields_3d'] = []
        drivers['input']['atmos']['fields_2d'] = []
        drivers['output']['atmos']['fields_3d'] = []
        drivers['output']['atmos']['fields_2d'] = []
    elif atmospheric_forcing == 'wind_only':
        iaf3d = drivers['input']['atmos']['fields_3d']
        iaf2d = drivers['input']['atmos']['fields_2d']
        oaf3d = drivers['output']['atmos']['fields_3d']
        oaf2d = drivers['output']['atmos']['fields_2d']
        drivers['input']['atmos']['fields_3d'] = [f for f in iaf3d if 'wind' in f]
        drivers['input']['atmos']['fields_2d'] = [f for f in iaf2d if 'wind' in f]
        drivers['output']['atmos']['fields_3d'] = [f for f in oaf3d if 'wind' in f]
        drivers['output']['atmos']['fields_2d'] = [f for f in oaf2d if 'wind' in f]
    elif atmospheric_forcing == 'thermo_only':
        iaf3d = drivers['input']['atmos']['fields_3d']
        iaf2d = drivers['input']['atmos']['fields_2d']
        oaf3d = drivers['output']['atmos']['fields_3d']
        oaf2d = drivers['output']['atmos']['fields_2d']
        drivers['input']['atmos']['fields_3d'] = [f for f in iaf3d if 'wind' not in f]
        drivers['input']['atmos']['fields_2d'] = [f for f in iaf2d if 'wind' not in f]
        drivers['output']['atmos']['fields_3d'] = [f for f in oaf3d if 'wind' not in f]
        drivers['output']['atmos']['fields_2d'] = [f for f in oaf2d if 'wind' not in f]
    
    # deaggregate variables
    atmos_3d, atmos_2d, ocean_3d, ocean_2d = deaggregate_systems(
        batch, 
        drivers['input']['atmos']['fields_3d'], 
        drivers['input']['atmos']['fields_2d'], 
        drivers['input']['ocean']['fields_3d'], 
        drivers['input']['ocean']['fields_2d'], 
    )
    
    # atmospheric forcing processing
    if atmospheric_forcing == 'persist':
        if atmos_3d is not None:
            # persist the same initial condition for the atmosphere
            first_elem_id = np.where(dates_ == date)[0][0]
            for i in range(first_elem_id, len(dates_)):
                atmos_3d[:,:,i] = atmos_3d[:,:,first_elem_id]
        if atmos_2d is not None:
            # persist the same initial condition for the atmosphere
            first_elem_id = np.where(dates_ == date)[0][0]
            for i in range(first_elem_id, len(dates_)):
                atmos_2d[:,:,i] = atmos_2d[:,:,first_elem_id]
    
    # get the input dates
    input_dates = [pd.to_datetime(file.split('/')[-1].split('.')[0]).strftime('%Y-%m-%d') for file in files]
    # create a data dictionary containing {date : data} items
    data_dict = create_data_dict((atmos_3d, atmos_2d, ocean_3d, ocean_2d), input_dates)
    # forecast dates contains the Initial Condition + all the forecasting dates
    forecast_dates = [date.strftime('%Y-%m-%d') for date in pd.date_range(date, date + pd.DateOffset(forecast_horizon), freq='1D')]
    # create a data dictionary of forecasts containing {date: None} items
    forecast_dict = {date: None for date in forecast_dates[1:]}
    ic_date = date
    if verbose: print(f'Forecasting the Time Series that starts from {ic_date}')
    if use_tqdm:
        iterator = tqdm(enumerate(forecast_dates[1:]), total=len(forecast_dates[1:]))
    else:
        iterator = enumerate(forecast_dates[1:])
    for i, date in iterator:
        # get the input dates to be used
        indates = [(pd.to_datetime(date) - pd.DateOffset(i+1, days=lead_time)).strftime('%Y-%m-%d') for i in reversed(range(insteps))]
        # get the output dates 
        outdates = [(pd.to_datetime(date) + pd.DateOffset(i, days=lead_time)).strftime('%Y-%m-%d') for i in range(outsteps)]
        if verbose: print(f'   predicting date {date} ({outdates})')
        # load the input data from the `forecast_dict` or the `data_dict`
        indata = load_input_data(indates, data_dict, forecast_dict, drivers, verbose)
        # stack the indata into a torch Tensor
        xa3d, xa2d, xo3d, xo2d = stack_data_from_dict(indata, dim=tdim)
        # time embedding processing
        if time_embedding == 'persist':
            # persist the same initial condition date
            t = torch.cat([torch.as_tensor([get_day_of_year(ic_date)]) for _ in indates])
        elif time_embedding == 'default':
            # get static time variable
            t = torch.cat([torch.as_tensor([get_day_of_year(date)]) for date in indates])
        t = t.unsqueeze(0)
        # move to device
        xa3d, xa2d, xo3d, xo2d, t = move_data_to_device((xa3d, xa2d, xo3d, xo2d, t), device)
        if verbose: print(f'   data moved to device')
        # forward pass
        if verbose: print(f'   forecasting')
        ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = model(
            x=(xa3d, xa2d, xo3d, xo2d), 
            t=t, r=i, 
            drivers=drivers, 
            out_attn=False, 
            agg_systems=False
        )
        # get the average of the ensemble predictions
        ya3d_mu_pred, ya2d_mu_pred, yo3d_mu_pred, yo2d_mu_pred = mean_prediction(
            (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
            is_ens = model.is_ens
        )
        # offload to cpu
        ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred = move_data_to_device(
            (ya3d_pred, ya2d_pred, yo3d_pred, yo2d_pred), 
            'cpu')
        ya3d_mu_pred, ya2d_mu_pred, yo3d_mu_pred, yo2d_mu_pred = move_data_to_device(
            (ya3d_mu_pred, ya2d_mu_pred, yo3d_mu_pred, yo2d_mu_pred), 
            'cpu')
        xa3d, xa2d, xo3d, xo2d, t = move_data_to_device(
            (xa3d, xa2d, xo3d, xo2d, t), 
            'cpu')
        if verbose: print(f'   data offloaded to cpu')
        # update the forecasting dictionary
        forecast_dict = update_forecast_dict(
            outdates, 
            (ya3d_mu_pred, ya2d_mu_pred, yo3d_mu_pred, yo2d_mu_pred), 
            forecast_dict, 
            tdim
        )
    # clean the forecast dictionary from additional keys
    to_remove_keys = []
    for key in forecast_dict.keys():
        if key not in forecast_dates:
            to_remove_keys.append(key)
    # for each key to remove
    for key in to_remove_keys:
        # remove the key from the dictionary
        del forecast_dict[key]
    forecast_dict.keys()
    # get the initial conditions
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = data_dict[ic_date.strftime('%Y-%m-%d')]
    # add time dimension to initial conditions
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = (
        ic_a3d.unsqueeze(tdim) if ic_a3d is not None else None, 
        ic_a2d.unsqueeze(tdim) if ic_a2d is not None else None, 
        ic_o3d.unsqueeze(tdim) if ic_o3d is not None else None, 
        ic_o2d.unsqueeze(tdim) if ic_o2d is not None else None, 
    )
    # stack forecasts together
    (fc_a3d, fc_a2d, fc_o3d, fc_o2d) = stack_data_from_dict(forecast_dict, dim=tdim)
    if verbose: print(f'  Forecasts stacked')

    # (torch.Size([1, 4, 1, 14, 181, 360]), torch.Size([1, 4, 1, 1, 181, 360]))
    # permute data to match the xarray
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = (
        rearrange(ic_a3d, 't v f z h w -> t f v z h w') if ic_a3d is not None else None, 
        rearrange(ic_a2d, 't v f z h w -> t f v z h w') if ic_a2d is not None else None, 
        rearrange(ic_o3d, 't v f z h w -> t f v z h w') if ic_o3d is not None else None, 
        rearrange(ic_o2d, 't v f z h w -> t f v z h w') if ic_o2d is not None else None, 
    )
    fc_a3d, fc_a2d, fc_o3d, fc_o2d = (
        rearrange(fc_a3d, 't v f z h w -> t f v z h w') if fc_a3d is not None else None, 
        rearrange(fc_a2d, 't v f z h w -> t f v z h w') if fc_a2d is not None else None, 
        rearrange(fc_o3d, 't v f z h w -> t f v z h w') if fc_o3d is not None else None, 
        rearrange(fc_o2d, 't v f z h w -> t f v z h w') if fc_o2d is not None else None, 
    )

    # rescale to original feature scale
    ic_a3d, ic_a2d, ic_o3d, ic_o2d = rescale((ic_a3d, ic_a2d, ic_o3d, ic_o2d), drivers, epsilon)
    fc_a3d, fc_a2d, fc_o3d, fc_o2d = rescale((fc_a3d, fc_a2d, fc_o3d, fc_o2d), drivers, epsilon)
    if verbose: print(f'  Forecasts rescaled to original feature scale')

    # return the list of forecasts of this time series
    return (ic_a3d, ic_a2d, ic_o3d, ic_o2d), (fc_a3d, fc_a2d, fc_o3d, fc_o2d)
