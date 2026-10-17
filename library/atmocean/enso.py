from dask.distributed import progress
import dask.delayed

from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import dask


NINO_COORD_REGIONS = {
    # taken from: https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
    '1+2': (0, -10, 270, 280),  # Niño 1+2 (0-10S, 90W-80W)
    '3': (5, -5, 210, 270),     # Niño 3 (5N-5S, 150W-90W)
    '3.4': (5, -5, 190, 240),   # Niño 3.4 (5N-5S, 170W-120W)
    '4': (5, -5, 200, 210),     # Niño 4 (5N-5S, 160E-150W)
}
NINO_LATLON_COORDS = {
    '1+2': '(0-10S, 90W-80W)', 
    '3': '(5N-5S, 150W-90W)', 
    '3.4': '(5N-5S, 170W-120W)', 
    '4': '(5N-5S, 160E-150W)', 
}

IOD_REGIONS = {
    'west': (10, -10, 50, 70), # (10S-10N, 50-70E)
    'east': (0, -10, 90, 110), # (0-10S, 90-110E)
}
IOD_LATLON_COORDS = {
    'west': '(10S-10N, 50-70E)', 
    'east': '(0-10S, 90-110E)', 
}

TNA_REGION = (25, 5, 305, 345) # Tropical North Atlantic region (5-25N, 55-15W)

ATL3_REGION = (3, -3, 340, 360) # North Atlantic Niño Region (3S-3N, 20W-0)

IOB_REGION = (20, -20, 40, 110) # Indian Ocean Basin Mode Region (20S-20N, 40E-110E)

PMM_REGIONS = {
    'north': (21, 7, 180, 252),
    'south': (-7, -21, 180, 252),
    'whole': (50, -30, 160, 280),
}
PMM_LATLON_COORDS = {
    'north': '(7–21°N, 180°–108°W)',
    'south': '(7–21°S, 180°–108°W)',
}

SEASONS = {
    'DJF' : [12, 1, 2], 
    'MAM' : [3, 4, 5], 
    'JJA' : [6, 7, 8], 
    'SON' : [9, 10, 11], 
    'all' : np.arange(1, 13).tolist(), 
}

def detrend(ds: xr.Dataset, dim: str, deg: int = 1):
    """
    This function de-trends the passed dataset dimension, 
    in order to remove any seasonal behavior
    """
    # polynomial fitting of the time-series
    p = ds.polyfit(dim = dim, deg = deg)
    # fit the dimension with the found coefficients
    fit = xr.polyval(ds[dim], p.polyfit_coefficients)
    # return the de-trended dataset
    return ds - fit

def dmi(
        ds: xr.Dataset,
        gt: xr.Dataset = None,
        clm: xr.Dataset = None,
        lat1: float = None, lat2: float = None,
        lon1: float = None, lon2: float = None,
        region: list[float] = None,
        remove_trend: bool = False,
        spatial_avg: bool = False,
        reverse_coords: bool = False,
        time_dim: str = 'time',
    ):
    if region is not None:
        if reverse_coords:
            lat2, lat1, lon1, lon2 = region
        else:
            lat1, lat2, lon1, lon2 = region
    # compute the climatology of the dataset
    if gt is not None and clm is None:
        clm = gt.groupby(f'{time_dim}.month').mean(f'{time_dim}')
    # select the subregion
    ds = ds.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
    clm = clm.sel(latitude=slice(lat1, lat2), longitude=slice(lon1, lon2))
    # subtract the monthly climatology from the dataset
    # an_ds = ds - clm
    an_ds = ds.groupby(f'{time_dim}.month') - clm
    # compute mean of lat and lon to get the indices
    if spatial_avg:
        weights = np.cos(np.deg2rad(an_ds.latitude))
        an_ds = an_ds.weighted(weights).mean(('latitude', 'longitude'))
    # de-trend to remove the trend of the time-series
    if remove_trend:
        an_ds = detrend(an_ds, dim=f'{time_dim}', deg=1)
    # return the dataset
    return an_ds

def compute_climatology(ds: xr.Dataset, field: str = None, client = None):
    # compute the climatology of the dataset
    if field is not None:
        clm = ds.isel(depth=0, level=0, missing_dims='ignore')[field].groupby('time.month').mean('time')
    else:
        clm = ds.isel(depth=0, level=0, missing_dims='ignore').groupby('time.month').mean('time')
    if client is not None:
        # start computation in the background
        x = client.compute(clm)
        # watch progress
        progress(x)
    return clm

def compute_box_mean_index(
    ds: xr.Dataset,
    clm: xr.Dataset,
    region: list[float],
    is_lat_reversed: bool = False,
):
    lat1, lat2, lon1, lon2 = region
    # subtract the monthly climatology from the dataset
    anom_ds = ds.groupby('time.month') - clm
    # get the slices of the lat and lon
    lat_slice = slice(lat2, lat1) if is_lat_reversed else slice(lat1, lat2)
    lon_slice = slice(lon1, lon2)
    # select the Index subregion
    anom_reg_ds = anom_ds.sel(latitude=lat_slice, longitude=lon_slice)
    # clm = clm.sel(latitude=lat_slice, longitude=lon_slice)
    # compute mean of lat and lon to get the indices
    weights = np.cos(np.deg2rad(anom_ds.latitude))
    index_ds = anom_reg_ds.weighted(weights).mean(('latitude', 'longitude'))
    return anom_ds, index_ds

# @dask.delayed
# def compute_ocean_nino_index(
#         ds: xr.Dataset,                 # input dataset
#         clm: xr.Dataset,                # climatology dataset
#         region: list[float] = None,     # target region/s for index computation
#         threshold: float = 0.5,         # threshold to detect positive or negative event
#         is_lat_reversed: bool = False,  # whether the latitude is reversed in the dataset
#         use_tqdm: bool = False,         # whether to use tqdm bar when computing the index
#     ):
#     # subtract the monthly climatology from the dataset
#     # get the slices of the lat and lon
#     # select the Nino subregion
#     # compute mean of lat and lon to get the indices
#     anom_ds, index_ds = compute_box_mean_index(
#         ds, clm, 
#         region=region, 
#         is_lat_reversed=is_lat_reversed
#     )
#     # calculate 3-month rolling mean of Nino 3.4 anomaly for the ONI
#     ocean_index = index_ds.rolling(time=3, center=True).mean().dropna(dim='time')
#     # compute boolean mask to find el nino and la nina events
#     ocean_index_bool = np.zeros(ocean_index.shape)
#     iterator = tqdm(range(len(ocean_index) - 4)) if use_tqdm else range(len(ocean_index) - 4)
#     for i in iterator:
#         if all(ocean_index[i:i+5] >= threshold):
#             ocean_index_bool[i:i+5] = 1
#         elif all(ocean_index[i:i+5] <= -threshold):
#             ocean_index_bool[i:i+5] = -1
#     # get the el nino and la nina months
#     positive_years = ocean_index[ocean_index_bool == 1].time.values
#     negative_years = ocean_index[ocean_index_bool == -1].time.values
#     # return the collection of data
#     return anom_ds, ocean_index, ocean_index_bool, positive_years, negative_years

@dask.delayed
def compute_ocean_nino_index(
    ds: xr.Dataset,
    clm: xr.Dataset,
    region: list[float] = None,
    threshold: float = 0.5,
    is_lat_reversed: bool = False,
    use_tqdm: bool = False,
):
    anom_ds, index_ds = compute_box_mean_index(
        ds, clm,
        region=region,
        is_lat_reversed=is_lat_reversed
    )
    ocean_index = index_ds.rolling(time=3, center=True).mean().dropna(dim='time')

    # porta i valori in numpy UNA volta sola: e' un array piccolo (poche decine
    # di mesi), non serve passare da xarray ad ogni confronto del loop
    values = ocean_index.values
    n = len(values)
    ocean_index_bool = np.zeros(n)

    iterator = tqdm(range(n - 4)) if use_tqdm else range(n - 4)
    for i in iterator:
        window = values[i:i + 5]
        if np.all(window >= threshold):
            ocean_index_bool[i:i + 5] = 1
        elif np.all(window <= -threshold):
            ocean_index_bool[i:i + 5] = -1

    positive_years = ocean_index.time.values[ocean_index_bool == 1]
    negative_years = ocean_index.time.values[ocean_index_bool == -1]

    return anom_ds, ocean_index, ocean_index_bool, positive_years, negative_years


def compute_indian_ocean_dipole_index(
        ds: xr.Dataset,                 # input dataset
        clm: xr.Dataset,                # climatology dataset
        regions: list[float] = None,    # target region/s for index computation
        is_lat_reversed: bool = False,  # whether the latitude is reversed in the dataset
        relax_constraints: bool = False,# whether to relax constraints on the IOD years
    ):
    # get IOD West and East regions
    west_region, east_region = regions
    # compute the West index
    anom_ds, west_index_ds = compute_box_mean_index(
        ds, clm, 
        region=west_region, 
        is_lat_reversed=is_lat_reversed
    )
    # compute the East index
    _, east_index_ds = compute_box_mean_index(
        ds, clm, 
        region=east_region, 
        is_lat_reversed=is_lat_reversed
    )
    # compute the IOD index (DMI) subtracting West and East
    iod_dmi = west_index_ds - east_index_ds
    iod_dmi_son = (
        iod_dmi
        .sel(time=iod_dmi.time.dt.season == "SON")
        .groupby("time.year")
        .mean("time")
    )
    if relax_constraints:
        sigma = iod_dmi_son.mean("year")
    else:
        sigma = iod_dmi_son.std("year")
    positive_years = iod_dmi_son > sigma
    negative_years = iod_dmi_son < -sigma
    return (
        anom_ds, 
        west_index_ds, east_index_ds, 
        iod_dmi, iod_dmi_son, 
        positive_years, negative_years, 
    )

def compute_atlantic_nino_index(
        ds: xr.Dataset,                # input dataset
        clm: xr.Dataset,               # climatology dataset
        region: list,                  # target region/s for index computation
        is_lat_reversed: bool = False, # whether the latitude is reversed in the dataset
        months: list[int] = [6,7,8],   # months on which we compute ATL3
    ):
    # compute the North Atlantic Nino 3 index
    anom_ds, index_ds = compute_box_mean_index(
        ds=ds, 
        clm=clm, 
        region=region, 
        is_lat_reversed=is_lat_reversed, 
    )
    # compute the seasonality of the index
    index_jjas_ds = (
        index_ds
        .sel(time=index_ds.time.dt.month.isin(months))
        .groupby("time.month")
        .mean("time")
    )
    # compute the threshold for positive and negative events
    sigma = index_jjas_ds.std("month")
    # get positive and negative years (useful for SST composite)
    positive_years = index_jjas_ds > sigma # index_jjas_ds > index_jjas_ds.quantile(0.8, "year") # sigma
    negative_years = index_jjas_ds < -sigma # index_jjas_ds < -index_jjas_ds.quantile(0.2, "year") # sigma
    return anom_ds, index_ds, index_jjas_ds, positive_years, negative_years

# @dask.delayed
def compute_tropical_north_atlantic_index(
        ds: xr.Dataset,                 # input dataset
        clm: xr.Dataset,                # climatology dataset
        region: list[float] = None,     # target region/s for index computation
        is_lat_reversed: bool = False,  # whether the latitude is reversed in the dataset
    ):
    # subtract the monthly climatology from the dataset
    # get the slices of the lat and lon
    # select the Nino subregion
    # compute mean of lat and lon to get the indices
    anom_ds, tna_index_ds = compute_box_mean_index(
        ds, clm, 
        region=region, 
        is_lat_reversed=is_lat_reversed
    )
    # compute the seasonality of the index
    tna_index_ds_seas = (
        tna_index_ds
        .groupby("time.month")
        .mean("time")
    )
    # compute the standard deviation of the TNA index
    sigma = tna_index_ds_seas.std("month")
    # find the positive and negative events
    tna_positive_years = tna_index_ds_seas > sigma
    tna_negative_years = tna_index_ds_seas < -sigma
    return anom_ds, tna_index_ds, tna_positive_years, tna_negative_years


def compute_bias_sst(
        anom_ds: xr.Dataset, 
        positive_years: list, 
        negative_years: list, 
        months: list, 
    ):
    sst_positive_values = (
        anom_ds
        .sel(time=positive_years)
        .where(anom_ds.time.dt.month.isin(months))
        .mean('time', skipna=True)
    )
    sst_negative_values = (
        anom_ds
        .sel(time=negative_years)
        .where(anom_ds.time.dt.month.isin(months))
        .mean('time', skipna=True)
    )
    if np.isnan(sst_positive_values).sum() == np.prod(sst_positive_values.shape):
        sst_positive_values = np.nan_to_num(sst_positive_values, nan=0.0)
    if np.isnan(sst_negative_values).sum() == np.prod(sst_negative_values.shape):
        sst_negative_values = np.nan_to_num(sst_negative_values, nan=0.0)
    return sst_positive_values - sst_negative_values


def compute_resampled_bias_sst(
        anom_ds: xr.Dataset, 
        positive_years: np.ndarray, 
        negative_years: np.ndarray, 
        months: list[int] = None, 
        resample: str = "year",
    ):
    """
    Compute SST spatial composites for positive and negative events.
    
    Parameters
    ----------
    anom_ds : xr.Dataset or xr.DataArray
        SST anomalies (time, lat, lon)
    positive_years : array-like
        Years of positive events
    negative_years : array-like
        Years of negative events
    months : list of int
        Months to select for the compositing (default SON)
    
    Returns
    -------
    sst_pos : xr.DataArray
        Composite SST for positive events
    sst_neg : xr.DataArray
        Composite SST for negative events
    """
    # Select Index months and then average over each year
    if months is not None:  sst_sel = anom_ds.where(anom_ds.time.dt.month.isin(months))
    else:                   sst_sel = anom_ds
    sst_resampled = sst_sel.groupby(f"time.{resample}").mean("time", skipna=True)
    # Compute the positive and negative composites
    sst_pos = sst_resampled.where(positive_years).mean(f"{resample}", skipna=True)
    sst_neg = sst_resampled.where(negative_years).mean(f"{resample}", skipna=True)
    # Compute the SST bias
    sst_diff = sst_pos.fillna(0.0) - sst_neg.fillna(0.0)
    return sst_diff, sst_pos, sst_neg


def compute_iod_bias_sst(
        anom_ds: xr.Dataset,
        positive_years: np.ndarray,
        negative_years: np.ndarray,
        months: list[int] = [9,10,11],
    ):
    return compute_resampled_bias_sst(
        anom_ds=anom_ds,
        positive_years=positive_years,
        negative_years=negative_years,
        months=months,
        resample='year',
    )


def compute_atl3_bias_sst(
        anom_ds: xr.Dataset, 
        positive_years: np.ndarray, 
        negative_years: np.ndarray, 
        months: list[int] = [6,7,8,9], 
    ):
    return compute_resampled_bias_sst(
        anom_ds=anom_ds,
        positive_years=positive_years,
        negative_years=negative_years,
        months=months,
        resample='month',
    )


def compute_tna_bias_sst(
        anom_ds: xr.Dataset, 
        positive_years: np.ndarray, 
        negative_years: np.ndarray, 
        months: list[int] = [6,7,8,9], 
    ):
    return compute_resampled_bias_sst(
        anom_ds=anom_ds,
        positive_years=positive_years,
        negative_years=negative_years,
        months=months,
        resample='month',
    )


def compute_isotherm_20C(
    ds: xr.Dataset, 
    max_depth: float = 300.0, 
):
    depth = ds.depth.data
    # Take the nan mask
    is_nan = ds.where(ds > 20.0).isnull()
    # Verify where we have at least one nan
    # has_nan = is_nan.any(dim="depth")
    # Get the first nan index. It will 
    # correspond to the index where we 
    # have temperature values below 20°C
    first_nan_idx = (
        is_nan
        .where(is_nan)        # False -> NaN
        .argmax(dim="depth")  # first True value from above
    ).compute()               # compute to avoid errors
    # Rename the variable for 
    # better code understanding
    below_20_ids = first_nan_idx
    above_20_ids = first_nan_idx - 1
    # Get the depth values at which we have
    # more than and less than 20°C
    depth_below_20 = depth[below_20_ids]
    depth_above_20 = depth[above_20_ids]
    # Get the temperature where we have less than 20°C
    temp_below_20 = ds.isel(depth=below_20_ids).values
    # Get the temperature where we have more than 20°C
    temp_above_20 = ds.isel(depth=above_20_ids).values
    # Find the depth at which we have exactly 20°C.
    # It will be the result of a linear interpolation
    # between the temperature values of the layer 
    # above 20°C and the layer below 20°C. It can be
    # computed using the formula:
    # Z20 = z1 + (20 - T1) * (z2 - z1) / (T2 - T1)
    Z20 = depth_above_20 + (20 - temp_above_20) * (depth_below_20 - depth_above_20) / (temp_below_20 - temp_above_20)
    # One last step: replace nan values with a 
    # higher depth value. We fix it to 300 m
    # that acts as a placeholder.
    Z20 = np.where(~np.isnan(Z20), Z20, max_depth)
    # get the coordinates of the dataset
    coords = dict(time=ds.time.data, latitude=ds.latitude.data, longitude=ds.longitude.data)
    if 'prediction_timedelta' in ds.coords.keys():
        coords = dict(time=ds.time.data, prediction_timedelta=ds.prediction_timedelta.data, latitude=ds.latitude.data, longitude=ds.longitude.data)
    else:
        coords = dict(time=ds.time.data, latitude=ds.latitude.data, longitude=ds.longitude.data)
    # convert the data to an xr.DataArray
    Z20 = xr.DataArray(data=Z20, coords=coords)
    return Z20

# def _isotherm_20c_np(temp, depth, max_depth):
#     """
#     temp: ndarray, l'ultima dimensione DEVE essere 'depth'
#     depth: ndarray 1D con i valori di profondità
#     """
#     below_threshold = temp <= 20.0
#     has_crossing = below_threshold.any(axis=-1)

#     below_idx = np.argmax(below_threshold, axis=-1)      # primo indice sotto i 20°C
#     above_idx = np.clip(below_idx - 1, 0, None)           # indice appena sopra

#     depth_below = depth[below_idx]
#     depth_above = depth[above_idx]

#     # gather vettorizzato al posto del doppio ds.isel(...).values
#     temp_below = np.take_along_axis(temp, below_idx[..., None], axis=-1)[..., 0]
#     temp_above = np.take_along_axis(temp, above_idx[..., None], axis=-1)[..., 0]

#     z20 = depth_above + (20.0 - temp_above) * (depth_below - depth_above) / (temp_below - temp_above)
#     z20 = np.where(has_crossing & ~np.isnan(z20), z20, max_depth)
#     return z20.astype(np.float32)


# def compute_isotherm_20C(ds: xr.Dataset, max_depth: float = 300.0):
#     depth = ds.depth.data  # coordinata piccola, ok tenerla in memoria

#     Z20 = xr.apply_ufunc(
#         _isotherm_20c_np,
#         ds,
#         input_core_dims=[['depth']],
#         kwargs={'depth': depth, 'max_depth': max_depth},
#         dask='parallelized',
#         output_dtypes=[np.float32],
#     )
#     return Z20


def compute_oceanic_sam(ds_ssh):
    # Compute Zonal Mean of SSH
    ssh_zonal = ds_ssh.mean(dim='longitude')
    # Compute the difference between 
    # 40°S and 60°S. A SAM+ sees a higher 
    # SSH at 40°S and a lower SSH at 60°S
    # Caused by the increase in the geostrophism
    sam_ssh = ssh_zonal.sel(latitude=-40, method='nearest') - \
              ssh_zonal.sel(latitude=-60, method='nearest')
    # Normalize
    return (sam_ssh - sam_ssh.mean()) / sam_ssh.std()


def compute_indian_ocean_basin_mode_index(
        ds: xr.Dataset,                # input dataset
        clm: xr.Dataset,               # climatology dataset
        region: list[float] = None,    # target region/s for index computation
        is_lat_reversed: bool = False, # whether to reverse the latitudes
    ):
    anom_ds, index_ds = compute_box_mean_index(
        ds=ds, clm=clm, region=region, is_lat_reversed=is_lat_reversed,
    )
    index_ds = (
        index_ds
        .groupby("time.month")
        .mean("time")
    )
    sigma = index_ds.mean('month')
    positive_years = index_ds > sigma
    negative_years = index_ds < -sigma
    return (
        anom_ds, index_ds, positive_years, negative_years,
    )


def compute_pacific_meridional_mode_index(
        ds: xr.Dataset,                 # input dataset
        clm: xr.Dataset,                # climatology dataset
        region: list[float] = None,     # target region (usa PMM_REGIONS['north'] o ['south'])
        is_lat_reversed: bool = False,  # se la latitudine è invertita nel dataset
        months: list[int] = None,       # mesi su cui calcolare la stagionalità
                                        # (None = indice mensile completo)
                                        # suggeriti: [3,4,5] per NPMM, [9,10,11] per SPMM
    ):
    """
    Calcola il Pacific Meridional Mode (PMM) index come media spaziale pesata
    delle anomalie SST nella regione target.

    Segue lo stesso pattern di compute_atlantic_nino_index / compute_tropical_north_atlantic_index.
    Per una stima più rigorosa (MCA su SST+vento), usa questa funzione come
    proxi box-mean, sufficiente per applicazioni S2S.

    Returns
    -------
    anom_ds       : xr.Dataset  – anomalie SST sull'intero dominio
    index_ds      : xr.DataArray – indice mensile PMM
    index_seas_ds : xr.DataArray – indice stagionale (media per anno)
                                   None se months=None
    positive_years : xr.DataArray – anni positivi (> 80° percentile)
    negative_years : xr.DataArray – anni negativi (< 20° percentile)
    """
    # Calcola anomalie e media spaziale nella regione
    anom_ds, index_ds = compute_box_mean_index(
        ds=ds,
        clm=clm,
        region=region,
        is_lat_reversed=is_lat_reversed,
    )

    index_seas_ds = (
        index_ds
        .sel(time=index_ds.time.dt.month.isin(months))
        .groupby("time.month")
        .mean("time")
    )

    sigma = index_seas_ds.std("month")
    pos_months = index_seas_ds > sigma
    neg_months = index_seas_ds < -sigma

    return anom_ds, index_ds, index_seas_ds, pos_months, neg_months


def compute_iob_bias_sst(
        anom_ds: xr.Dataset,
        positive_years: np.ndarray,
        negative_years: np.ndarray,
        months: list[int] = [i for i in range(1,13)],
    ):
    return compute_resampled_bias_sst(
        anom_ds=anom_ds, 
        positive_years=positive_years, 
        negative_years=negative_years, 
        months=months, 
        resample="month",
    )


def compute_pmm_bias_sst(
        anom_ds: xr.Dataset,
        positive_years: np.ndarray,
        negative_years: np.ndarray,
        months: list[int] = None,       # [3,4,5] per NPMM, [9,10,11] per SPMM
    ):
    """
    Spatial composite SST per eventi PMM positivi vs negativi.
    Wrapper identico a compute_atl3_bias_sst / compute_iod_bias_sst.
    """
    return compute_resampled_bias_sst(
        anom_ds=anom_ds,
        positive_years=positive_years,
        negative_years=negative_years,
        months=months,
        resample="month",
    )


# def monthly_resample_with_prediction_timedelta(ds:xr.Dataset):
#     """
#     Shifts the `time` coordinate of each lead time to its month 
#     of validity (IC + prediction_timedelta), then monthly resamples
    
#     In this way, the groupby('time.month') - clm always subtracts the
#     correct climatology
#     """
#     # Convert prediction_timedelta in whole days
#     lead_days = (
#         ds['prediction_timedelta']
#         .values
#         .astype('timedelta64[D]')
#         .astype(int)
#     )
#     # For each lead time, shift time and resample
#     resampled = []
#     for i, days in enumerate(lead_days):
#         ds_lead = ds.isel(prediction_timedelta=i).copy()
#         ds_lead['time'] = (
#             pd.to_datetime(ds_lead['time'].values) + pd.DateOffset(days=int(days))
#         )
#         ds_lead = ds_lead.resample({'time': '1M'}, skipna=True).mean(dim='time')
#         resampled.append(ds_lead)
    
#     # Aggregate along prediction_timedelta dimension
#     return xr.concat(resampled, dim='prediction_timedelta')


# def monthly_groupby_with_prediction_timedelta(ds: xr.Dataset):
#     """
#     Shifts the dataset at the validity month for each lead time
#     and then groups by time.
#     """
#     lead_days = (
#         ds['prediction_timedelta']
#         .values
#         .astype('timedelta64[D]')
#         .astype(int)
#     )
#     resampled = []
#     for i, days in enumerate(lead_days):
#         ds_lead = ds.isel(prediction_timedelta=i).copy()
#         ds_lead['time'] = (
#             pd.to_datetime(ds_lead['time'].values) + pd.DateOffset(days=int(days))
#         )
#         ds_lead = ds_lead.groupby('time.month').mean('time')
#         resampled.append(ds_lead)
#     return xr.concat(resampled, dim='prediction_timedelta')
# pip install flox --break-system-packages   (una tantum, xarray lo rileva da solo)

def _shift_and_reduce(ds, lead_days, reducer):
    results = []
    for i, days in enumerate(lead_days):
        ds_lead = ds.isel(prediction_timedelta=i)  # no .copy(): e' gia' una view lazy
        ds_lead = ds_lead.assign_coords(
            time=pd.to_datetime(ds_lead['time'].values) + pd.DateOffset(days=int(days))
        )
        results.append(reducer(ds_lead))
    return xr.concat(results, dim='prediction_timedelta')


def monthly_resample_with_prediction_timedelta(ds: xr.Dataset, time_chunk: int = 100):
    # spezza il chunk unico su 'time' per parallelizzare il resample (flox lavora a blocchi)
    ds = ds.chunk({'time': time_chunk})
    lead_days = ds['prediction_timedelta'].values.astype('timedelta64[D]').astype(int)
    return _shift_and_reduce(
        ds, lead_days,
        reducer=lambda d: d.resample(time='1M', skipna=True).mean(dim='time'),
    )


def monthly_groupby_with_prediction_timedelta(ds: xr.Dataset, time_chunk: int = 100):
    ds = ds.chunk({'time': time_chunk})
    lead_days = ds['prediction_timedelta'].values.astype('timedelta64[D]').astype(int)
    return _shift_and_reduce(
        ds, lead_days,
        reducer=lambda d: d.groupby('time.month').mean('time'),
    )

def compute_tke(ds:xr.DataArray) -> xr.DataArray:
    """
    Calcola la Total Kinetic Energy superficiale.

    TKE = 0.5 * (u² + v²)

    La media temporale è calcolata sull'asse `time` (le IC mensili),
    per ogni prediction_timedelta separatamente — coerente con
    il tuo framework S2S.

    Returns
    -------
    tke : xr.DataArray (time, prediction_timedelta, lat, lon) [m²/s²]
    """
    tke = 0.5 * (ds['uo'] ** 2 + ds['vo'] ** 2)
    tke.attrs['units'] = 'm²/s²'
    tke.attrs['long_name'] = 'Total Kinetic Energy'
    return tke


def compute_eke(ds:xr.DataArray) -> xr.DataArray:
    """
    Calcola la Eddy Kinetic Energy superficiale.

    EKE = 0.5 * (u² + v²)
    dove u' = u - mean(u, dim=time), v' = v - mean(v, dim=time)

    La media temporale è calcolata sull'asse `time` (le IC mensili),
    per ogni prediction_timedelta separatamente — coerente con
    il tuo framework S2S.

    Returns
    -------
    eke : xr.DataArray (time, prediction_timedelta, lat, lon) [m²/s²]
    """
    uo_prime = ds['uo'] - ds['uo'].mean(dim='time', skipna=True)
    vo_prime = ds['vo'] - ds['vo'].mean(dim='time', skipna=True)
    eke = 0.5 * (uo_prime ** 2 + vo_prime ** 2)
    eke.attrs['units'] = 'm²/s²'
    eke.attrs['long_name'] = 'Eddy Kinetic Energy'
    return eke

# def compute_eke_bias(
#         eke_forecast: xr.DataArray,     # (time, prediction_timedelta, lat, lon)
#         eke_observed: xr.DataArray,     # (time, lat, lon) — da reanalysis
#         lead_days: list[int] = [0, 30, 60],
#     ) -> xr.DataArray:
#     """
#     Bias EKE = mean_time(EKE_forecast) - mean_time(EKE_obs)
#     per lead time specifici.

#     Returns
#     -------
#     bias : xr.DataArray (lead_day, lat, lon)
#     """
#     obs_mean = eke_observed.mean(dim='time')
#     maps = []
#     for day in lead_days:
#         fcs_mean = eke_forecast.isel(prediction_timedelta=day).mean(dim='time')
#         maps.append(fcs_mean - obs_mean)
#     return xr.concat(maps, dim=xr.Variable('lead_day', lead_days))


def compute_brier_score(
        forecasts: xr.DataArray,     # (time, prediction_timedelta, lat, lon) — valori in [0,1]
        threshold: float = 0.15,        # soglia WMO per binarizzare l'osservazione
        time_dim: str = 'time',
    ) -> xr.DataArray:
    """
    Calcola il Brier Score spaziale per ogni lead time.

    BS = mean_time[ (SIC_forecast - I(SIC_obs > threshold))² ]

    Returns
    -------
    bs : xr.DataArray (prediction_timedelta, lat, lon)
         Brier Score medio nel tempo per ogni cella e lead time
    """
    # Binarizza l'osservazione
    binary = (forecasts.isel(prediction_timedelta=0) > threshold).astype(float)  # (time, lat, lon)
    brier_score = ( ( forecasts - binary ) ** 2).mean(dim=time_dim)
    # binary = (forecasts > threshold).astype(float)  # (time, prediction_timedelta, lat, lon)
    # brier_score = ( ( binary - binary.isel(prediction_timedelta=0) ) ** 2).mean(dim=time_dim)
    return brier_score


def compute_brier_skill_score(
        brier_score,
        brier_score_clm,
    ) -> xr.DataArray:
    """
    Brier Skill Score: BSS = 1 - BS/BS_ref
    BSS > 0 → meglio della climatologia
    BSS = 1 → forecast perfetto
    BSS < 0 → peggio della climatologia

    bs_ref si calcola passando la climatologia mensile di SIC come forecast.
    """
    return 1.0 - (brier_score / brier_score_clm.where(brier_score_clm > 0))


def compute_ocean_heat(delta_level, temperature):
    rho = 1026 #kg/m^3
    c_p = 3990 #J/(kg K)
    # convert temperature to Kelvin
    temperature = temperature + 273.15
    if len(temperature.shape) == 6:     # we have ensemble dimension as first
        weighted_temperature = delta_level.depth.values[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * temperature.data
        weighted_temperature_sum = weighted_temperature.sum(axis=3)
    elif len(temperature.shape) == 5:   # we do not have the ensemble dimension
        weighted_temperature = delta_level.depth.values[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis] * temperature.data
        weighted_temperature_sum = weighted_temperature.sum(axis=2)
    heat = weighted_temperature_sum * rho * c_p
    return heat

def compute_ocean_heat_optim(delta_level, temperature):
    rho = np.float32(1026)   # kg/m^3
    c_p = np.float32(3990)   # J/(kg K)
    offset = np.float32(273.15)
    temperature = temperature + offset
    weights = delta_level.depth.astype(np.float32)
    heat = (temperature * weights).sum(dim='depth', skipna=False) * rho * c_p
    return heat


