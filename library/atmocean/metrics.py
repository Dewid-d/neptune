# from atmocean.losses import CRPS

from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import torch
import dask
import os


def create_empty_metrics_dataset(
    ds: xr.Dataset, 
):
    # get the dataset coordinates
    coords = ds.coords
    # remove ensemble member key if present
    if 'ensemble_member' in coords.keys(): del coords['ensemble_member']
    # create rmse dataset
    metric = xr.Dataset(coords = {key : value for key, value in coords.items() if key != 'time'})
    # fill with empty data
    for field in ds.data_vars.keys():
        dims = [dim for dim in ds[field].dims if dim != 'time']
        time_id = [i for i, dim in enumerate(ds[field].dims) if dim=='time'][0]
        shape = [x for i, x in enumerate(ds[field].shape) if i != time_id]
        metric[field] = (dims, np.empty(shape = shape, dtype = np.float32))
    # chunk the dataset
    chunks = {
        'latitude': ds.latitude.shape[0], 
        'longitude': ds.longitude.shape[0], 
        'depth': 1, 
        # 'level': 1, 
        'prediction_timedelta' : 1
    }
    if 'depth' not in ds.dims:
        del chunks['depth']
    metric = metric.chunk(chunks=chunks)
    return metric

# @dask.delayed
def compute_metrics_over_dataset(
    ds: xr.Dataset,
    forecast_horizon: int,
    metric_name: str = 'rmse',
    clm: xr.Dataset = None,
    device: str = 'cpu',
):
    available_metrics_list = ['rmse', 'bias', 'crps', 'mae', 'acc']
    if metric_name not in available_metrics_list: 
        raise ValueError(f'The passed metric was not recognized. Given: {metric_name}. Choose one of : {available_metrics_list}')
    metrics_ds_list = []
    # get the initial conditions
    initial_conditions = ds.isel(prediction_timedelta=0)
    if isinstance(forecast_horizon, list):
        iterable = tqdm(range(len(forecast_horizon)))
    else:
        iterable = tqdm(range(forecast_horizon + 1))
    for i in iterable:
        # get the forecasts at index `i`
        forecasts = ds.isel(prediction_timedelta=i)
        # update the forecasts timestamp
        forecasts['time'] = ds.time.data + ds.prediction_timedelta.data[i]
        # compute the chosen metric of the forecasts
        if metric_name == 'rmse':
            iterable.set_description('RMSE')
            computed_metrics = np.sqrt(((forecasts - initial_conditions)**2).mean(dim='time', skipna=True))
        elif metric_name == 'bias':
            iterable.set_description('BIAS')
            computed_metrics = (forecasts - initial_conditions).mean(dim='time', skipna=True)
        elif metric_name == 'mae':
            iterable.set_description('MAE')
            computed_metrics = np.abs(forecasts - initial_conditions).mean(dim='time', skipna=True)
        elif metric_name == 'crps':
            iterable.set_description('CRPS')
            diff_dates = (initial_conditions - forecasts).time.data

            has_ensemble = 'ensemble_member' in ds.dims
            if has_ensemble:
                forecasts = forecasts.chunk({'time': -1, 'ensemble_member': -1})
                initial_conditions = initial_conditions.chunk({'time': -1, 'ensemble_member': -1})
            else:
                forecasts = forecasts.chunk({'time': -1})
                initial_conditions = initial_conditions.chunk({'time': -1})

            time_mask = forecasts.time.isin(diff_dates)
            forecasts = forecasts.isel(time=time_mask).persist()
            initial_conditions = initial_conditions.isel(time=time_mask).persist()

            field_names = list(forecasts.data_vars)

            with torch.no_grad():
                if has_ensemble:
                    iterable.set_postfix({'crps-impl': 'ensemble'})
                    # stack: (field, time, lat, lon)
                    ic_stack = torch.as_tensor(np.stack(
                        [initial_conditions.isel(ensemble_member=0)[f].values for f in field_names]
                    )).to(device)
                    fc_stack = torch.as_tensor(np.stack(
                        [forecasts[f].mean(dim='ensemble_member', skipna=True).values for f in field_names]
                    )).to(device)
                    std_stack = torch.as_tensor(np.stack(
                        [forecasts[f].std(dim='ensemble_member', skipna=True).values for f in field_names]
                    )).to(device)
                    out_dims = forecasts[field_names[0]].dims[2:]
                else:
                    iterable.set_postfix({'crps-impl': 'default'})
                    ic_stack = torch.as_tensor(np.stack(
                        [initial_conditions[f].values for f in field_names]
                    )).to(device)
                    fc_stack = torch.as_tensor(np.stack(
                        [forecasts[f].values for f in field_names]
                    )).to(device)
                    std_stack = torch.zeros_like(fc_stack).to(device) + 1e-2
                    out_dims = forecasts[field_names[0]].dims[1:]

                crps_all = CRPS(y=ic_stack, mu=fc_stack, std_dev=std_stack)
                metric_all = torch.nanmean(crps_all, dim=1).cpu().numpy()  # (field, lat, lon)

                del ic_stack, fc_stack, std_stack, crps_all
                if device != 'cpu':
                    torch.cuda.empty_cache()

            field_data = {
                f: (out_dims, metric_all[i]) for i, f in enumerate(field_names)
            }
            computed_metrics = xr.Dataset(field_data, coords={k: v for k, v in ds.coords.items() if k != 'time'})
        # elif metric_name == 'acc':
        #     if clm is None:
        #         raise ValueError('Climatology must be passed if the computed metric is the Anomaly Correlation Coefficient')
        #     fcs_anom = (forecasts - clm)
        #     obs_anom = (initial_conditions - clm)
        #     field_data = {}
        #     for field in forecasts.data_vars:
        #         cov = xr.cov(fcs_anom[field], obs_anom[field], dim='time')
        #         fcs_std = fcs_anom[field].std(dim='time', skipna=True)
        #         obs_std = obs_anom[field].std(dim='time', skipna=True)
        #         acc = cov / (fcs_std * obs_std)
        #         field_data[field] = (acc.dims, acc.data)
        #     computed_metrics = xr.Dataset(field_data, coords={k: v for k, v in ds.coords.items() if k != 'time'})
        elif metric_name == 'acc':
            if clm is None:
                raise ValueError('Climatology must be passed if the computed metric is the Anomaly Correlation Coefficient')
            fcs_anom = (forecasts - clm)
            obs_anom = (initial_conditions - clm)
            field_list = []
            for field in forecasts.data_vars:
                cov = xr.cov(fcs_anom[field], obs_anom[field], dim='time')
                fcs_std = fcs_anom[field].std(dim='time', skipna=True)
                obs_std = obs_anom[field].std(dim='time', skipna=True)
                acc = (cov / (fcs_std * obs_std)).rename(field)
                field_list.append(acc)
            computed_metrics = xr.merge(field_list)
            # computed_metrics = xr.merge(field_list, join='override', compat='override')
        metrics_ds_list.append(computed_metrics)
    return metrics_ds_list


def update_metrics_from_dataset_list(
    metrics_ds_list: list[xr.Dataset], 
    metrics_ds: xr.Dataset, 
    ds: xr.Dataset, 
):
    for i,mtr in enumerate(metrics_ds_list):
        metrics_ds.loc[{'prediction_timedelta': ds.prediction_timedelta.data[i]}] = mtr
    return metrics_ds


def compute_and_store_metrics(
    ds: xr.Dataset,             # dataset to use for computing the metric
    forecast_horizon: int,      # forecasting horizon of the dataset
    dst: str,                   # destination file to store the computed metric
    metric_name: str = 'rmse',  # name of the metric to be computed
    clm: xr.Dataset = None,     # dataset to use as climatology (pass only when computing ACC)
    device: str = 'cpu',        # device on which we compute the metric (CPU default, use only for CRPS)
):
    if dst is None or not os.path.exists(dst):
        # create an empty dataset to store the rmse
        metrics_ds = create_empty_metrics_dataset(ds)
        # compute the mean squared error over the dataset
        metrics_ds_list = compute_metrics_over_dataset(
            ds = ds, 
            forecast_horizon = forecast_horizon, 
            metric_name = metric_name, 
            clm = clm, 
            device = device, 
        )
        # update the dataset with the forecasts
        metrics_ds = update_metrics_from_dataset_list(
            metrics_ds_list = metrics_ds_list, 
            metrics_ds = metrics_ds, 
            ds = ds
        )
        if dst is not None:
            # store to zarr archive
            return metrics_ds.to_zarr(dst, mode='w', compute=False, zarr_format=2)
        else:
            return metrics_ds
    return xr.open_zarr(dst)

def get_persistence_ds(src: str, ds: xr.Dataset):
    if not os.path.exists(src):
        # get the coordinates shapes for dataset chunking
        n_lats, n_lons = ds['latitude'].shape[0], ds['longitude'].shape[0]
        n_levels = ds['level'].shape[0]
        n_depths = ds['depth'].shape[0]
        persistence_ds = ds.isel(prediction_timedelta=0).copy()
        persistence_ds = persistence_ds.expand_dims(dim={'prediction_timedelta': ds.prediction_timedelta}).ffill(dim='prediction_timedelta')
        persistence_ds.chunk(chunks={'time': 1, 'prediction_timedelta':len(ds.prediction_timedelta), 'depth':n_depths, 'level':n_levels, 'latitude': n_lats, 'longitude': n_lons})
        persistence_ds = persistence_ds.map(lambda x: x.astype('float32') if x.dtype.kind == 'f' else x)
        persistence_ds.to_zarr(src)
    return xr.open_zarr(src, chunks={})





def compute_crps_group(
        forecasts_paired,
        initial_conditions,
        field_list,
        device: str = 'cpu',
    ):
    fc = forecasts_paired[field_list].to_array(dim='field')
    ic = initial_conditions[field_list].to_array(dim='field')

    fc_t = torch.as_tensor(fc.values, device=device)
    ic_t = torch.as_tensor(ic.values, device=device)

    ens_dim = fc.dims.index('ensemble_member')
    mu = torch.mean(fc_t, dim=ens_dim)
    std = torch.std(fc_t, dim=ens_dim) + 1e-7
    y = ic_t.unsqueeze(2).expand_as(mu)

    crps = CRPS(y=y, mu=mu, std_dev=std).cpu().numpy()

    # esclusione POSIZIONALE, non per nome: evita ambiguità se un nome
    # compare più volte o l'ordine non è quello atteso
    dims_list = list(fc.dims)
    del dims_list[ens_dim]
    out_dims = tuple(dims_list)

    # sanity check: numero dimensioni deve combaciare con lo shape reale
    assert len(out_dims) == crps.ndim, f"Mismatch dims {out_dims} vs shape {crps.shape}"

    del fc_t, ic_t, mu, std, y
    if device != 'cpu':
        torch.cuda.empty_cache()

    return crps, out_dims

def compute_and_store_crps_metrics(
        ds: xr.Dataset,
        dst,
        tmp_dir,
        fields_3d,
        fields_2d,
        valid_start_date,
        valid_end_date,
        lead_times,
        batch_size: int = 30,
    ):
    if os.path.exists(dst):
        return xr.open_zarr(dst)
    all_fields = fields_3d + fields_2d
    valid_dates = pd.date_range(valid_start_date, valid_end_date, freq='1D')
    date_offsets = pd.to_timedelta(lead_times, unit='days')
    n_lead = len(lead_times)
    n_batches = int(np.ceil(len(valid_dates) / batch_size))
    tmp_dst = os.path.join(tmp_dir, '')
    first_write = not __import__('os').path.exists(tmp_dst)

    for b in tqdm(range(n_batches), desc='batch'):
        batch_dates = valid_dates[b * batch_size: (b + 1) * batch_size]
        n_valid_b = len(batch_dates)

        # date di init necessarie SOLO per questo batch
        fc_dates = batch_dates.values[:, None] - date_offsets.values[None, :]  # (n_valid_b, n_lead)
        unique_init_dates = pd.DatetimeIndex(np.unique(fc_dates.ravel()))

        # carica in memoria solo la finestra di questo batch
        ds_sel = ds[all_fields].sel(time=unique_init_dates).load()
        time_index = pd.Index(ds_sel.time.values)

        time_pos = time_index.get_indexer(fc_dates.ravel()).reshape(n_valid_b, n_lead)
        lead_pos = np.broadcast_to(np.arange(n_lead), (n_valid_b, n_lead))

        time_idx = xr.DataArray(time_pos, dims=('valid_time', 'lead'))
        lead_idx = xr.DataArray(lead_pos, dims=('valid_time', 'lead'))

        forecasts_paired = ds_sel.isel(time=time_idx, prediction_timedelta=lead_idx)

        ic_pos = time_index.get_indexer(batch_dates)
        initial_conditions = ds_sel.isel(
            time=xr.DataArray(ic_pos, dims='valid_time'),
            prediction_timedelta=0,
            ensemble_member=0,
        )

        crps_3d, dims_3d = compute_crps_group(forecasts_paired, initial_conditions, fields_3d)
        crps_2d, dims_2d = compute_crps_group(forecasts_paired, initial_conditions, fields_2d)

        spatial_coords_3d = {d: forecasts_paired[fields_3d[0]].coords[d] for d in dims_3d if d not in ('field', 'valid_time', 'lead')}
        spatial_coords_2d = {d: forecasts_paired[fields_2d[0]].coords[d] for d in dims_2d if d not in ('field', 'valid_time', 'lead')}

        batch_result = xr.Dataset(
            {
                **{f'crps_{f}': (dims_3d[1:], crps_3d[i]) for i, f in enumerate(fields_3d)},
                **{f'crps_{f}': (dims_2d[1:], crps_2d[i]) for i, f in enumerate(fields_2d)},
            },
            coords={'valid_time': batch_dates, 'lead': lead_times, **spatial_coords_3d, **spatial_coords_2d},
        )

        # scrittura incrementale su zarr: prima batch crea, le successive fanno append su valid_time
        if first_write:
            batch_result.to_zarr(tmp_dst, mode='w', zarr_format=2)
            first_write = False
        else:
            batch_result.to_zarr(tmp_dst, mode='a', append_dim='valid_time', zarr_format=2)

        del ds_sel, forecasts_paired, initial_conditions, batch_result
    
    # pre-process the zarr dataset and store into the final directory
    crps_ds = xr.open_zarr(tmp_dst)
    crps_ds = crps_ds.rename_dims({'valid_time':'time', 'lead':'prediction_timedelta'}).rename_vars({'valid_time':'time', 'lead':'prediction_timedelta'})
    rename_vars_dict = {f'crps_{field}':field for field in all_fields}
    crps_ds = crps_ds.rename_vars(rename_vars_dict)
    crps_ds['prediction_timedelta'] = pd.to_timedelta(crps_ds['prediction_timedelta'].values, unit='days')
    crps_ds = crps_ds.mean(dim='time', skipna=True)
    try:
        crps_ds.to_zarr(dst, zarr_format=2)
    except:
        print('Error occurred during dataset storing. Store by yourself')
        return crps_ds
    # remove temporary dataset
    # tmp_dst