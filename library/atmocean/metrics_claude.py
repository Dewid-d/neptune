from atmocean.losses import CRPS

import xarray as xr
import numpy as np
import torch
import dask
import os


def create_empty_metrics_dataset(ds: xr.Dataset):
    coords = dict(ds.coords)
    coords.pop('ensemble_member', None)
    coords.pop('time', None)
    metric = xr.Dataset(coords=coords)
    for field in ds.data_vars.keys():
        dims = tuple(d for d in ds[field].dims if d != 'time')
        shape = tuple(s for d, s in zip(ds[field].dims, ds[field].shape) if d != 'time')
        metric[field] = (dims, np.empty(shape=shape, dtype=np.float32))
    chunk_kwargs = {}
    for dim in ('latitude', 'longitude', 'depth', 'level', 'prediction_timedelta'):
        if dim in metric.dims and dim in ds.dims:
            chunk_kwargs[dim] = ds.sizes[dim]
    metric = metric.chunk(chunk_kwargs)
    return metric


def _align_forecasts_to_truth(ds: xr.Dataset, n_steps: int):
    """
    ds ha dims (time=data di inizializzazione, prediction_timedelta=lead, ...).
    La IC (prediction_timedelta=0) di un giorno X è la verità osservata in quel giorno.
    Un forecast inizializzato a t con lead L è valido al tempo (t + L):
    quindi va confrontato con la IC del giorno (t + L), non con quella di t.
    """
    truth = ds.isel(prediction_timedelta=0)  # dims: (time, ...) — 'time' = data reale/osservata
    forecasts = ds.isel(prediction_timedelta=slice(0, n_steps)).rename({'time': 'init_time'})

    valid_time = forecasts['init_time'] + forecasts['prediction_timedelta']  # dims: (init_time, prediction_timedelta)

    # reindex (non sel!) così le valid_time fuori range (a fine dataset) diventano NaN
    # invece di sollevare KeyError
    truth_aligned = truth.reindex(time=valid_time)  # dims: (init_time, prediction_timedelta, ...)

    return forecasts, truth_aligned


def _crps_for_timedelta(fc_i, truth_i, device):
    """CRPS per un singolo prediction_timedelta, già allineato a valid_time."""
    # scarta le date senza verità disponibile (bordo destro del dataset)
    truth_i = truth_i.dropna(dim='init_time', how='all')
    fc_i = fc_i.sel(init_time=truth_i.init_time)

    computed_metrics = xr.Dataset(coords={k: v for k, v in fc_i.coords.items() if k != 'init_time'})
    for field in fc_i.data_vars:
        if 'ensemble_member' in fc_i[field].dims:
            ic_field = torch.as_tensor(truth_i[field].values, dtype=torch.float32, device=device)
            fc_field = torch.as_tensor(
                fc_i[field].mean(dim='ensemble_member', skipna=True).values, dtype=torch.float32, device=device
            )
            std_dev = torch.as_tensor(
                fc_i[field].std(dim='ensemble_member', skipna=True).values, dtype=torch.float32, device=device
            )
            out_dims = fc_i[field].dims[2:]
        else:
            ic_field = torch.as_tensor(truth_i[field].values, dtype=torch.float32, device=device)
            fc_field = torch.as_tensor(fc_i[field].values, dtype=torch.float32, device=device)
            std_dev = torch.zeros_like(fc_field) + 1e-2
            out_dims = fc_i[field].dims[1:]

        field_metrics = torch.nanmean(CRPS(y=ic_field, mu=fc_field, std_dev=std_dev), axis=0).cpu().numpy()
        computed_metrics[field] = (out_dims, field_metrics.astype(np.float32))
    return computed_metrics


def compute_metrics_over_dataset(
    ds: xr.Dataset,
    forecast_horizon: int,
    metric_name: str = 'rmse',
    clm: xr.Dataset = None,
    device: str = 'cpu',
):
    available_metrics_list = ['rmse', 'bias', 'crps', 'mae', 'acc']
    if metric_name not in available_metrics_list:
        raise ValueError(
            f'The passed metric was not recognized. Given: {metric_name}. '
            f'Choose one of : {available_metrics_list}'
        )

    n_steps = len(forecast_horizon) if isinstance(forecast_horizon, list) else forecast_horizon + 1

    # ---- rmse / bias / mae: vettorizzati, allineati a valid_time, nessun loop ----
    if metric_name in ('rmse', 'bias', 'mae'):
        forecasts, truth_aligned = _align_forecasts_to_truth(ds, n_steps)
        diff = forecasts.astype('float32') - truth_aligned
        if metric_name == 'rmse':
            return np.sqrt((diff ** 2).mean(dim='init_time', skipna=True))
        elif metric_name == 'bias':
            return diff.mean(dim='init_time', skipna=True)
        else:  # mae
            return np.abs(diff).mean(dim='init_time', skipna=True)

    # ---- acc: stesso allineamento, poi cov/std per field ----
    elif metric_name == 'acc':
        if clm is None:
            raise ValueError(
                'Climatology must be passed if the computed metric is the Anomaly Correlation Coefficient'
            )
        forecasts, truth_aligned = _align_forecasts_to_truth(ds, n_steps)
        # NOTA: assicurati che `clm` sia indicizzata in modo compatibile con la valid_time
        # (es. per day-of-year), altrimenti va anch'essa riallineata con lo stesso schema.
        fcs_anom = forecasts - clm
        obs_anom = truth_aligned - clm
        computed_metrics = xr.Dataset(coords=forecasts.coords)
        for field in forecasts.data_vars:
            cov = xr.cov(fcs_anom[field], obs_anom[field], dim='init_time')
            fcs_std = fcs_anom[field].std(dim='init_time', skipna=True)
            obs_std = obs_anom[field].std(dim='init_time', skipna=True)
            computed_metrics[field] = (cov.dims, (cov / (fcs_std * obs_std)).data)
        return computed_metrics

    # ---- crps: stesso allineamento, poi un delayed per lead time (richiede torch) ----
    elif metric_name == 'crps':
        forecasts, truth_aligned = _align_forecasts_to_truth(ds, n_steps)
        tasks = []
        for i in range(n_steps):
            fc_i = forecasts.isel(prediction_timedelta=i)
            truth_i = truth_aligned.isel(prediction_timedelta=i)
            tasks.append(dask.delayed(_crps_for_timedelta)(fc_i, truth_i, device))
        results = dask.compute(*tasks)
        return list(results)


def update_metrics_from_dataset_list(
    metrics_ds_list: list,
    metrics_ds: xr.Dataset,
    ds: xr.Dataset,
):
    for i, mtr in enumerate(metrics_ds_list):
        metrics_ds.loc[{'prediction_timedelta': ds.prediction_timedelta.data[i]}] = mtr
    return metrics_ds


def compute_and_store_metrics(
    ds: xr.Dataset,
    forecast_horizon: int,
    dst: str,
    metric_name: str = 'rmse',
    clm: xr.Dataset = None,
    device: str = 'cpu',
):
    if dst is None or not os.path.exists(dst):
        result = compute_metrics_over_dataset(
            ds=ds, forecast_horizon=forecast_horizon, metric_name=metric_name, clm=clm, device=device,
        )

        if metric_name == 'crps':
            metrics_ds = create_empty_metrics_dataset(ds)
            metrics_ds = update_metrics_from_dataset_list(result, metrics_ds, ds)
        else:
            metrics_ds = result.astype('float32')

        if dst is not None:
            metrics_ds.to_zarr(dst, mode='w', compute=True)
        else:
            return metrics_ds
    return xr.open_zarr(dst)


def get_persistence_ds(src: str, ds: xr.Dataset):
    if not os.path.exists(src):
        n_lats, n_lons = ds['latitude'].shape[0], ds['longitude'].shape[0]
        n_levels = ds['level'].shape[0]
        n_depths = ds['depth'].shape[0]
        persistence_ds = ds.isel(prediction_timedelta=0).astype('float32').copy()
        persistence_ds = persistence_ds.expand_dims(
            dim={'prediction_timedelta': ds.prediction_timedelta}
        ).ffill(dim='prediction_timedelta')
        persistence_ds = persistence_ds.chunk(chunks={
            'time': 1,
            'prediction_timedelta': len(ds.prediction_timedelta),
            'depth': n_depths,
            'level': n_levels,
            'latitude': n_lats,
            'longitude': n_lons,
        })
        persistence_ds.to_zarr(src)
    return xr.open_zarr(src, chunks={})