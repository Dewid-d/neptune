# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# ----------------------------------------------------------------------------

from atmocean.macros import ATMOS_3D_IDS, ATMOS_2D_IDS, OCEAN_3D_IDS, OCEAN_2D_IDS

from math import radians, sin, cos, asin, sqrt
from typing import AnyStr, Union, List
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import logging
import shutil
import torch
import math
import toml
import os



def log_data(title, *args):
    string = f'{title} '
    for arg in args:
        string += str(arg.shape) + " "
    print(string)


def log_message(log_fns, msg):
    for log_fn in log_fns:
        log_fn(msg)


def fields_dict_to_toml(src: Union[xr.Dataset, AnyStr], dst: AnyStr):
    """ Writes a toml file containing fields attributes from a xr.Dataset. """
    if isinstance(src, xr.Dataset):
        ds = src
    elif isinstance(src, str):
        ds = xr.open_dataset(src)
    else:
        raise AttributeError
    vars_dict = dict()
    for var in ds.data_vars:
        vars_dict[var] = ds[var].attrs
    with open(dst, "w") as toml_file:
        toml.dump(vars_dict, toml_file)
        
        
def round_robin_split(l: List, n_proc: int):
    """ Split a list into n_proc sublist in a round robin fashion. """
    lists = [[] for _ in range(n_proc)]
    i = 0
    for elem in l:
        lists[i].append(elem)
        i = (i + 1) % n_proc
    return lists


def seed_everything(seed):
    import torch
    # python seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy seed
    np.random.seed(seed)
    # pytorch seed
    torch.manual_seed(seed)
    print(f'Execution seeded successfully with seed {seed}')


def get_padded_shape(shape, patch_size):
    sh, ps = np.array(shape), np.array(patch_size)
    mod = sh % ps
    return tuple(sh + ps - np.where(mod == 0, ps, mod))


def padding(x: torch.Tensor, 
            rs: tuple[int, int, int], # reference shape to pad from x
            ps: tuple[int, int, int], # patch size wrt pad the reference shape
            crop: bool=False, 
            ch_last: bool=True):
    if len(x.shape) == 6:
        dims1 = (0,5,1,2,3,4)
        dims2 = (0,2,3,4,5,1)
    elif len(x.shape) == 5:
        dims1 = (0,4,1,2,3) 
        dims2 = (0,2,3,4,1)
    elif len(x.shape) == 4:
        dims1 = (0,3,1,2)
        dims2 = (0,2,3,1)
    s, p = np.array(rs), np.array(ps)
    mod = s % p
    pad_amount = p - np.where(mod == 0, p, mod)
    left_pad, right_pad = np.array([0]*pad_amount.shape[0]), pad_amount
    if crop:
        left_pad, right_pad = -left_pad, -right_pad
    padding = np.stack([left_pad, right_pad], axis=1).astype(np.int16)
    padding = tuple(padding[::-1].flatten())
    if ch_last:
        x = x.permute(dims=dims1)
        x = F.pad(x, padding)
        x = x.permute(dims=dims2)
        return x
    else:
        return F.pad(x, padding)


def reflect_rotate_pad(x: torch.Tensor, pad_sequence: int):
    """Applies a padding with reflection and rotation over the Poles."""
    has_batch = True if len(pad_sequence) < x.dim() * 2 - 2 else False
    for i, pad in enumerate(reversed(pad_sequence)):
        if pad == 0: continue
        dim = i//2
        dim = dim + 2 if has_batch else dim + 1
        D, Dd = x.size(dim), x.size(dim+1)
        if i % 2 == 0: indices = np.arange(0,pad) # take from 0 to pad
        else: indices = reversed(np.arange(D - pad, D)) # take from D to D - pad
        indices = [j for j in indices]
        add_pad = torch.index_select(x, dim=dim, index=torch.tensor(indices, device=x.device)) # select only the index
        add_pad = torch.roll(add_pad, shifts=(Dd//2,), dims=(dim+1,)) # roll the tensor by half
        if i % 2 == 0: add_pad = torch.flip(add_pad, dims=(dim,)) # flip to the opposite side
        if i % 2 == 0: x = torch.cat((add_pad, x), dim=dim)
        else: x = torch.cat((x, add_pad), dim=dim)
    return x


# must only pad H and W
def physical_padding(
        x: torch.Tensor, 
        patch_size: tuple = (32,32), 
        pattern: str = None, 
        crop: bool = False, 
        ori_shape: tuple = None, 
    ) -> torch.Tensor:
    """
    Applies a padding that preserves the consistency of the Earth shape. 
    The consistency is obtained by padding over the Poles with with a 
    reflect and 180° rotationfollowed by a circular padding over the 
    Longitudes to keep continuity on the boundary of the map.
    """
    x_shape = x.shape
    in_shape = ori_shape or x.shape[-2:]
    out_shape = get_padded_shape(shape=in_shape, patch_size=patch_size)
    iH, iW = in_shape
    oH, oW = out_shape
    dH, dW = oH - iH, oW - iW
    W1_pad, W2_pad = int(np.floor(dW / 2)), int(np.ceil(dW / 2))
    H1_pad, H2_pad = int(np.floor(dH / 2)), int(np.ceil(dH / 2))
    assert len(x_shape) == len(pattern.split(' '))
    assert pattern.split(' ')[-2:] == ['h', 'w'], f'Input data must be of shape ... x H x W. Given {pattern}. X {x.shape}'
    kwargs = {p: o for p, o in zip(pattern.split(' '), x_shape)}
    # knowing that h and w are in the last places
    spl_pat = pattern.split(' ')
    to_pattern = '(' + ' '.join(spl_pat[:-2]) + ') ' + ' '.join(spl_pat[-2:])
    x = rearrange(x, f'{pattern} -> {to_pattern}', **kwargs) # K x H x W with K = prod of other dims
    if crop:
        x = F.pad(x, pad=(-W2_pad, -W1_pad, -H2_pad, -H1_pad))
        kwargs['h'], kwargs['w'] = x.shape[-2], x.shape[-1]
        x = rearrange(x, f'{to_pattern} -> {pattern}', **kwargs)
        return x
    # apply reflection and rotation padding
    x = reflect_rotate_pad(x, (0, 0, H1_pad, H2_pad))
    # apply circular paddding
    x = F.pad(x, pad=(W1_pad, W2_pad), mode='circular')
    # update h and w with the new shapes
    kwargs['h'], kwargs['w'] = x.shape[-2], x.shape[-1]
    # recover to the original data shape
    x = rearrange(x, f'{to_pattern} -> {pattern}', **kwargs)
    return x


def get_num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_class_name(instance):
    return str(instance.__class__).split(".")[-1].split("'")[0]


def is_xarray_zarr_datastore_consistent(path):
    try:
        # test consistency of the dataset
        ds = xr.open_zarr(path)
        ds.close()
        return True
    except:
        # remove inconsistent file
        shutil.rmtree(path)
        # log
        logging.info(f'   Downloaded zarr is not consistent. Removed.')
        return False


def is_xarray_netcdf_consistent(path):
    try:
        # test consistency of the dataset
        ds = xr.open_dataset(path)
        ds.close()
        return True
    except:
        # remove inconsistent file
        os.remove(path)
        # log
        logging.info(f'   Downloaded dataset is not consistent. Removed.')
        return False


def aggregate_ocean_variables(vars_3d, vars_2d):
    # B x Z x H x W
    thetao_oras, so_oras, uo_oras, vo_oras = vars_3d[:,0], vars_3d[:,1], vars_3d[:,2], vars_3d[:,3]
    # B x C x H x W
    zos_oras, mlotst_oras, siconc_oras, sithick_oras = vars_2d[:,0:1], vars_2d[:,1:2], vars_2d[:,2:3], vars_2d[:,3:]
    # stack together variables
    ocean_fields = torch.cat([thetao_oras, so_oras, uo_oras, vo_oras, zos_oras, mlotst_oras, siconc_oras, sithick_oras], dim=1)
    return ocean_fields, thetao_oras, so_oras, uo_oras, vo_oras, zos_oras, mlotst_oras, siconc_oras, sithick_oras


def aggregate_atmos_variables(vars_3d, vars_2d):
    # B x Z x H x W
    geopotential, temperature, u_component_of_wind, v_component_of_wind, specific_humidity, vertical_velocity = vars_3d[:,0], vars_3d[:,1], vars_3d[:,2], vars_3d[:,3], vars_3d[:,4], vars_3d[:,5]
    # B x C x H x W
    mean_sea_level_pressure, sea_ice_cover, sea_surface_temperature, surface_pressure = vars_2d[:,0:1], vars_2d[:,1:2], vars_2d[:,2:3], vars_2d[:,3:]
    # stack together variables
    ocean_fields = torch.cat([geopotential, temperature, u_component_of_wind, v_component_of_wind, specific_humidity, vertical_velocity, mean_sea_level_pressure, sea_ice_cover, sea_surface_temperature, surface_pressure], dim=1)
    return ocean_fields, geopotential, temperature, u_component_of_wind, v_component_of_wind, specific_humidity, vertical_velocity, mean_sea_level_pressure, sea_ice_cover, sea_surface_temperature, surface_pressure


def day_month_of_year_embed(fname, shape):
    '''Day and Month of the Year (DMoY) co-sinusoidal embedding'''
    date = os.path.basename(fname).split('.npy')[0]
    _, month, day = int(date[:4]), int(date[4:6]), int(date[-2:])
    # DoM embedding
    d_sin_enc = torch.full(shape, np.sin(2 * np.pi * day/31.0), dtype=torch.float32)
    d_cos_enc = torch.full(shape, np.cos(2 * np.pi * day/31.0), dtype=torch.float32)
    # MoY embedding
    m_sin_enc = torch.full(shape, np.sin(2 * np.pi * month/12.0), dtype=torch.float32)
    m_cos_enc = torch.full(shape, np.cos(2 * np.pi * month/12.0), dtype=torch.float32)
    return torch.stack([d_sin_enc, d_cos_enc, m_sin_enc, m_cos_enc])


def day_of_year_embed(fname, shape):
    '''Day and Month of the Year (DMoY) co-sinusoidal embedding'''
    date = os.path.basename(fname).split('.npy')[0]
    date = pd.to_datetime(date)
    day_of_year = pd.Period(date, freq='d').day_of_year
    # DoY embedding
    sin_enc = torch.full(shape, np.sin(2 * np.pi * day_of_year/366.0), dtype=torch.float32)
    cos_enc = torch.full(shape, np.cos(2 * np.pi * day_of_year/366.0), dtype=torch.float32)
    return torch.stack([sin_enc, cos_enc])


def get_day_of_year(fname):
    '''Day and Month of the Year (DMoY)'''
    date = os.path.basename(fname).split('.npy')[0]
    date = pd.to_datetime(date)
    day_of_year = pd.Period(date, freq='d').day_of_year
    return day_of_year


def deaggregate_systems(x: torch.Tensor, atmos_3d_vars: list[str], atmos_2d_vars: list[str], ocean_3d_vars: list[str], ocean_2d_vars: list[str]):
    # get each separate system
    atmos_3d = torch.stack([x[:,ATMOS_3D_IDS[v]] for v in atmos_3d_vars], dim=1).permute(0,1,3,2,4,5) if len(atmos_3d_vars) != 0 else None # B x V x T x Z x H x W
    atmos_2d = torch.stack([x[:,ATMOS_2D_IDS[v]] for v in atmos_2d_vars], dim=1).permute(0,1,3,2,4,5) if len(atmos_2d_vars) != 0 else None # B x V x T x 1 x H x W
    ocean_3d = torch.stack([x[:,OCEAN_3D_IDS[v]] for v in ocean_3d_vars], dim=1).permute(0,1,3,2,4,5) if len(ocean_3d_vars) != 0 else None # B x V x T x Z x H x W
    ocean_2d = torch.stack([x[:,OCEAN_2D_IDS[v]] for v in ocean_2d_vars], dim=1).permute(0,1,3,2,4,5) if len(ocean_2d_vars) != 0 else None # B x V x T x 1 x H x W
    return atmos_3d, atmos_2d, ocean_3d, ocean_2d


def aggregate_systems(atmos_3d: torch.Tensor, atmos_2d: torch.Tensor, ocean_3d: torch.Tensor, ocean_2d: torch.Tensor, is_ens = False):
    # assume this shape B V T Z H W
    if is_ens: pattern = 'b v t z h w e -> b (v z) t h w e'
    else:      pattern = 'b v t z h w -> b (v z) t h w'
    x = torch.cat([rearrange(s, pattern) for s in (atmos_3d, atmos_2d, ocean_3d, ocean_2d) if s is not None], dim=1)
    return x


def deaggregate_system_embeddings(x, res1, res2, res3, res4, reshape: bool = True):
    r1, r2, r3, r4 = res1[1] if res1 is not None else 0, res2[1] if res2 is not None else 0, res3[1] if res3 is not None else 0, res4[1] if res4 is not None else 0
    x1 = x[:,:,0 : r1] if r1 != 0 else None
    x2 = x[:,:,r1 : r1 + r2] if r2 != 0 else None
    x3 = x[:,:,r1 + r2 : r1 + r2 + r3] if r3 != 0 else None
    x4 = x[:,:,r1 + r2 + r3 : r1 + r2 + r3 + r4] if r4 != 0 else None
    if reshape:
        # xs = [rearrange(x, 'b t z h w d -> b (t z h w) d') for x in xs]
        x1 = rearrange(x1, 'b t z h w d -> b (t z h w) d') if x1 is not None else None
        x2 = rearrange(x2, 'b t z h w d -> b (t z h w) d') if x2 is not None else None
        x3 = rearrange(x3, 'b t z h w d -> b (t z h w) d') if x3 is not None else None
        x4 = rearrange(x4, 'b t z h w d -> b (t z h w) d') if x4 is not None else None
    return x1, x2, x3, x4


def get_system_ids(drivers, inout: str = 'input'):
    k = 0
    atmos_3d_ids = {
        var: [j + i * len(drivers['levels']) + k for j in range(len(drivers['levels']))] 
        for i, var in enumerate(drivers[inout]['atmos']['fields_3d'])
    }

    k += len(atmos_3d_ids.keys()) * len(drivers['levels'])
    atmos_2d_ids = {
        var: [i + k] 
        for i, var in enumerate(drivers[inout]['atmos']['fields_2d'])
    }
    
    k += len(atmos_2d_ids.keys())
    ocean_3d_ids = {
        var: [j + i * len(drivers['depths']) + k for j in range(len(drivers['depths']))] 
        for i, var in enumerate(drivers[inout]['ocean']['fields_3d'])
    }
    
    k += len(ocean_3d_ids.keys()) * len(drivers['depths'])
    ocean_2d_ids = {
        var: [i + k] 
        for i, var in enumerate(drivers[inout]['ocean']['fields_2d'])
    }
    return atmos_3d_ids, atmos_2d_ids, ocean_3d_ids, ocean_2d_ids


def deaggregate_systems_by_drivers(x, drivers, inout: str = 'input'):
    atmos_fields_3d = drivers[inout]['atmos']['fields_3d']
    atmos_fields_2d = drivers[inout]['atmos']['fields_2d']
    ocean_fields_3d = drivers[inout]['ocean']['fields_3d']
    ocean_fields_2d = drivers[inout]['ocean']['fields_2d']
    atmos_3d_ids, atmos_2d_ids, ocean_3d_ids, ocean_2d_ids = get_system_ids(drivers, inout)
    atmos_3d = torch.stack([x[:,atmos_3d_ids[v]] for v in atmos_fields_3d], dim=1).permute(0,1,3,2,4,5) if len(atmos_fields_3d) != 0 else None # B x V x T x Z x H x W
    atmos_2d = torch.stack([x[:,atmos_2d_ids[v]] for v in atmos_fields_2d], dim=1).permute(0,1,3,2,4,5) if len(atmos_fields_2d) != 0 else None # B x V x T x 1 x H x W
    ocean_3d = torch.stack([x[:,ocean_3d_ids[v]] for v in ocean_fields_3d], dim=1).permute(0,1,3,2,4,5) if len(ocean_fields_3d) != 0 else None # B x V x T x Z x H x W
    ocean_2d = torch.stack([x[:,ocean_2d_ids[v]] for v in ocean_fields_2d], dim=1).permute(0,1,3,2,4,5) if len(ocean_fields_2d) != 0 else None # B x V x T x 1 x H x W
    return atmos_3d, atmos_2d, ocean_3d, ocean_2d


def deaggregate_system_into_variables_dict(x, drivers, inout: str = 'input'):
    atmos_fields_3d = drivers[inout]['atmos']['fields_3d']
    atmos_fields_2d = drivers[inout]['atmos']['fields_2d']
    ocean_fields_3d = drivers[inout]['ocean']['fields_3d']
    ocean_fields_2d = drivers[inout]['ocean']['fields_2d']
    atmos_3d_ids, atmos_2d_ids, ocean_3d_ids, ocean_2d_ids = get_system_ids(drivers, inout)
    atmos_3d_vars = {v : x[:,atmos_3d_ids[v]].permute(0,2,1,3,4) for v in atmos_fields_3d} if len(atmos_fields_3d) != 0 else None
    atmos_2d_vars = {v : x[:,atmos_2d_ids[v]].permute(0,2,1,3,4) for v in atmos_fields_2d} if len(atmos_fields_2d) != 0 else None
    ocean_3d_vars = {v : x[:,ocean_3d_ids[v]].permute(0,2,1,3,4) for v in ocean_fields_3d} if len(ocean_fields_3d) != 0 else None
    ocean_2d_vars = {v : x[:,ocean_2d_ids[v]].permute(0,2,1,3,4) for v in ocean_fields_2d} if len(ocean_fields_2d) != 0 else None
    return atmos_3d_vars, atmos_2d_vars, ocean_3d_vars, ocean_2d_vars


def realign_drivers_dict(drivers: dict):
    drvs = {}
    try:
        drvs['depths'] = [float(f[0]) for f in drivers['depths']]
        drvs['levels'] = [int(f[0]) for f in drivers['levels']]
        drvs['input'] = {}
        drvs['output'] = {}
        drvs['input']['atmos'] = {}
        drvs['input']['atmos']['fields_3d'] = [f[0] for f in drivers['input']['atmos']['fields_3d']]
        drvs['input']['atmos']['fields_2d'] = [f[0] for f in drivers['input']['atmos']['fields_2d']]
        drvs['input']['ocean'] = {}
        drvs['input']['ocean']['fields_3d'] = [f[0] for f in drivers['input']['ocean']['fields_3d']]
        drvs['input']['ocean']['fields_2d'] = [f[0] for f in drivers['input']['ocean']['fields_2d']]
        drvs['output']['atmos'] = {}
        drvs['output']['atmos']['fields_3d'] = [f[0] for f in drivers['output']['atmos']['fields_3d']]
        drvs['output']['atmos']['fields_2d'] = [f[0] for f in drivers['output']['atmos']['fields_2d']]
        drvs['output']['ocean'] = {}
        drvs['output']['ocean']['fields_3d'] = [f[0] for f in drivers['output']['ocean']['fields_3d']]
        drvs['output']['ocean']['fields_2d'] = [f[0] for f in drivers['output']['ocean']['fields_2d']]       
        return drvs
    except Exception as e:
        # print(f'Exception occurred: {e}')
        return drivers


def clean_state_dict_checkpoint_wrapped_module_from_keys(state: dict):
    """
    Check if `_checkpoint_wrapped_module` is in the keys of the state dictionary
    and returns a cleaned version
    """
    new_state = state.copy()
    for key in state.keys():
        if '_checkpoint_wrapped_module' in key:
            new_key = ''.join(key.split('_checkpoint_wrapped_module.'))
            new_state[new_key] = new_state[key]
            del new_state[key]
    return new_state


def clean_state_dict_forward_module_from_keys(state_dict: dict):
    """
    Check if `_forward_module` is in the keys of the state dictionary
    and returns a cleaned version
    """
    clean_state_dict = {}
    for key in state_dict.keys():
        # if `_forward_module` is present
        # we have to clean the name of the key
        if '_forward_module' in key:
            clean_key = ''.join(key.split('_forward_module.'))
            clean_state_dict[clean_key] = state_dict[key]
        # otherwise we can add the key as it it
        else:
            clean_state_dict[key] = state_dict[key]
    return clean_state_dict


def compute_psd_sht(
        x, 
        sht, 
        Lmax: int, 
        R_earth: float = 6371.0, # earth radius in km
    ):
    # Spherical Harmonic Transform
    alm = sht(x)
    # Compute the radial spectrum
    alm_np = alm.squeeze(0).cpu().numpy()
    power = np.zeros(Lmax)
    # compute the PSD for each wavelength
    for l in range(Lmax):
        power[l] = np.abs(alm_np[l,0])**2
        if l > 0:
            m_end = min(l+1, alm_np.shape[1])  # avoid out of bound
            power[l] += 2 * np.nansum(np.abs(alm_np[l,1:m_end])**2)
    ell = np.arange(1, Lmax+1)
    lambda_km = 2 * np.pi * R_earth / ell
    psd = power
    return lambda_km, psd


def isotropic_noise(
    leading_shape: tuple[int, ...],
    lmax: int,  # length of the ℓ axis expected by isht
    mmax: int,  # length of the m axis expected by isht
    isht: torch.Tensor, 
    device: torch.device,
) -> torch.Tensor:
    """
    Implementation taken from https://ai2-climate-emulator.readthedocs.io/en/2025.10.0/_modules/fme/ace/registry/stochastic_sfno.html#NoiseConditionedSFNO
    """
    # --- draw independent N(0,1) parts --------------------------------------
    coeff_shape = (*leading_shape, lmax, mmax)
    real = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    imag = torch.randn(coeff_shape, dtype=torch.float32, device=device)
    imag[..., :, 0] = 0.0  # m = 0 ⇒ purely real

    # m > 0: make Re and Im each N(0,½)  → |a_{ℓ m}|² has variance 1
    sqrt2 = math.sqrt(2.0)
    real[..., :, 1:] /= sqrt2
    imag[..., :, 1:] /= sqrt2

    # --- global scale that makes Var[T(θ,φ)] = 1 ---------------------------
    scale = math.sqrt(4.0 * math.pi) / lmax  # (Unsöld theorem ⇒ L = lmax)
    alm = (real + 1j * imag) * scale

    return isht(alm)


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if (abs(lon2-lon1) > 300):
        if (lon2 > 300):
           lon2= lon2-360
        else:
           lon1= lon1-360
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000 # Radius of earth in meter. Use 3956 for miles
    return c * r


def compute_dx_dy(ds: xr.Dataset):
    latitude, longitude = ds.latitude.values, ds.longitude.values
    ny, nx = len(latitude), len(longitude)
    dy = np.zeros((ny, nx))
    dx = np.zeros((ny, nx))
    for iy,y in enumerate(latitude[0 : (ny-1)]):
        for ix,x in enumerate(longitude[0 : (nx-1)]):
            dy[iy,ix] = haversine(longitude[ix], latitude[iy], longitude[ix], latitude[iy+1]) # calculation of the distance in y direction
            dx[iy,ix] = haversine(longitude[ix], latitude[iy], longitude[ix+1], latitude[iy]) # calculation of the distance in x direction
    ##  closure on last lines
    # x axis
    dx[:,nx-1] = dx[:,nx-2]
    dx[ny-1,:] = dx[ny-2,:]
    dx[ny-1,nx-1] = dx[ny-2,nx-2]
    # y axis
    dy[:,nx-1] = dy[:,nx-2]
    dy[ny-1,:] = dy[ny-2,:]
    dy[ny-1,nx-1] = dy[ny-2,nx-2]
    return dx, dy


def clean_forecasts(ds):
    ds = ds.isel(time=slice(1, None))
    return ds


def remove_ocean_variables_suffix(ds: xr.Dataset):
    vars_rename = dict()
    for var in ds.data_vars:
        vars_rename[var] = '_'.join(var.split('_')[:-1])
    ds = ds.rename_vars(vars_rename)
    return ds


# this is a new version of the prediction timedelta alignment
def align_prediction_timedelta(ds):
    lead_days = (
        ds['prediction_timedelta']
        .values
        .astype('timedelta64[D]')
        .astype(int)
    )
    offset = xr.DataArray(
        lead_days.astype('timedelta64[D]'),
        dims='prediction_timedelta',
        coords={'prediction_timedelta': ds['prediction_timedelta']},
    )
    # broadcasting time (dim 'time') + offset (dim 'prediction_timedelta')
    # produce esattamente la stessa coordinata 2D che il loop+concat costruiva a mano
    return ds.assign_coords(time=ds['time'] + offset)


def align_prediction_timedelta_optim(ds: xr.Dataset):
    lead_days = (
        ds['prediction_timedelta'].values.astype('timedelta64[D]').astype(int) - 1
    )
    base_time = pd.to_datetime(ds['time'].values)
    # (prediction_timedelta, time) shiftato, costruito in numpy: nessun concat/align
    valid_time = base_time.values[None, :] + lead_days[:, None].astype('timedelta64[D]')
    ds = ds.assign_coords(valid_time=(('prediction_timedelta', 'time'), valid_time))
    return ds
