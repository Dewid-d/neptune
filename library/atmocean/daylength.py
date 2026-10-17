from datetime import date, datetime
import pandas as pd
import numpy as np


def solar_declination(dates, hour=0.0):
    """
    Compute the solar declination angle (degrees) for given date(s) and hour.

    Parameters
    ----------
    dates : date, datetime, or list/array of those
        Calendar date(s). Time-of-day is ignored unless you pass `hour`.
    hour : float or array-like
        Hours since midnight (UTC), may be fractional (e.g. 13.5 = 13:30).
        Must be length 1 or same length as `dates`.

    Returns
    -------
    np.ndarray
        Solar declination angle(s) in degrees.

    Examples
    --------
    >>> solar_declination(date(2024, 3, 20), hour=3.1)   # near 0°
    >>> solar_declination(date(2024, 6, 21))              # near +23.44°
    >>> solar_declination([date(2024, 3, 20), date(2024, 6, 21)])
    """
    # Normalise inputs to arrays
    if isinstance(dates, (date, datetime)):
        dates = [dates]
    dates = list(dates)
    hour  = np.atleast_1d(np.asarray(hour, dtype=float))

    n_date = len(dates)
    n_hour = len(hour)
    if n_hour != 1 and n_hour != n_date:
        raise ValueError("`hour` must have length 1 or the same length as `dates`.")

    day   = np.array([d.day   for d in dates], dtype=float)
    month = np.array([d.month for d in dates], dtype=float)
    year  = np.array([d.year  for d in dates], dtype=float)

    # Jan/Feb adjustment
    mask = month <= 2
    year[mask]  -= 1
    month[mask] += 12

    # Julian Day (Gregorian calendar)
    A  = np.floor(year / 100)
    B  = 2 - A + np.floor(A / 4)
    JD = (np.floor(365.25 * (year + 4716))
          + np.floor(30.6001 * (month + 1))
          + day + B - 1524.5
          + hour / 24)

    # Days since J2000.0
    n = JD - 2451545.0

    # Mean longitude (deg)
    L = (280.46 + 0.9856474 * n) % 360
    # Mean anomaly (deg)
    g = (357.528 + 0.9856003 * n) % 360
    # Ecliptic longitude (deg)
    lam = (L + 1.915 * np.sin(np.radians(g))
             + 0.020 * np.sin(np.radians(2 * g))) % 360
    # Obliquity (deg)
    eps = 23.439 - 0.0000004 * n
    # Declination (deg)
    delta = np.degrees(np.arcsin(np.sin(np.radians(eps)) * np.sin(np.radians(lam))))

    return delta


def day_length(dates, lat_deg, hour=12.0):
    """
    Compute astronomical day length (photoperiod) in hours.

    Parameters
    ----------
    dates : date, datetime, or list/array of those
    lat_deg : float or array-like
        Latitude in degrees (N positive, S negative).
    hour : float or array-like
        Hour used for solar declination (default 12 = local noon).

    Returns
    -------
    np.ndarray
        Day length in hours.

    Examples
    --------
    >>> day_length(date(2026, 6, 21), lat_deg=44.69)  # Bedford Basin, NS
    """
    if isinstance(dates, (date, datetime)):
        dates = [dates]
    dates = list(dates)

    lat_deg = np.atleast_1d(np.asarray(lat_deg, dtype=float))
    hour    = np.atleast_1d(np.asarray(hour,    dtype=float))

    n_date = len(dates)
    if not np.all(np.isfinite(lat_deg)):
        raise ValueError("`lat_deg` must be finite.")
    if np.any((lat_deg < -90) | (lat_deg > 90)):
        raise ValueError("`lat_deg` must be between -90 and 90.")
    if len(lat_deg) not in (1, n_date):
        raise ValueError("`lat_deg` must have length 1 or the same length as `dates`.")
    if len(hour) not in (1, n_date):
        raise ValueError("`hour` must have length 1 or the same length as `dates`.")

    delta = np.radians(solar_declination(dates, hour))
    phi   = np.radians(lat_deg)

    x = np.clip(-np.tan(phi) * np.tan(delta), -1, 1)
    return (24 / np.pi) * np.arccos(x)


def day_length_vector(
        dates,
        lats=None,
        lons=None,
        grids=None,
        hour=12.0,
    ):
    """
    Compute astronomical day length (photoperiod) in hours.

    Parameters
    ----------
    dates : date, datetime, or list/array of those
    grid : float or array-like
        Latitude in degrees (N positive, S negative).
    hour : float or array-like
        Hour used for solar declination (default 12 = local noon).

    Returns
    -------
    np.ndarray
        Day length in hours.

    Examples
    --------
    >>> day_length(date(2026, 6, 21), lat_deg=44.69)  # Bedford Basin, NS
    """
    assert (lats is not None and lons is not None) or (grids is not None), f"`lats`, `lons` and `grid` are None. Must provide `lats` and `lons` or `grids`"
    # we do not need longrid in this case
    if grids is not None:
        _, latgrid = grids
    else:
        _, latgrid = np.meshgrid(lons, lats)
    if isinstance(dates, (date, datetime)):
        dates = [dates]
    dates = list(dates)
    
    hour = np.atleast_1d(np.asarray(hour, dtype=float))
    
    n_date = len(dates)
    # NOTE: ADD ASSERTIONS
    delta = np.radians(solar_declination(dates, hour))
    phi   = np.radians(latgrid)
    
    x = np.clip(-np.tan(phi)[:,:,np.newaxis] * np.tan(delta)[np.newaxis, np.newaxis], -1, 1)
    res = ((24 / np.pi) * np.arccos(x))
    return res

def from_date_to_doy(date):
    date = pd.to_datetime(date)
    day_of_year = pd.Period(date, freq='d').day_of_year
    return day_of_year

def from_doy_to_date(doy: int):
    date = pd.to_datetime((364*2+1) + doy, unit='D')
    return date
