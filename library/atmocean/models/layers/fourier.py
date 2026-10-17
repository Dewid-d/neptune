# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# Customized from Aurora Code
# ----------------------------------------------------------------------------

import torch.nn as nn
import numpy as np
import torch
import math



class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.

radius_earth = 6378137 / 1000
"""float: Radius of the earth in kilometers."""

def area(polygon: torch.Tensor) -> torch.Tensor:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    This function is a PyTorch port of the PyPI package `area`. In particular, it is heavily
    inspired by the following file:

        https://github.com/scisco/area/blob/9d9549d6ebffcbe4bffe11b71efa2d406d1c9fe9/area/__init__.py

    Args:
        polygon (:class:`torch.Tensor`): Polygon of the shape `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of the polygon, and 2
            concatenates first latitudes and then longitudes. The polygon does not have be closed.

    Returns:
        :class:`torch.Tensor`: Area in square kilometers.
    """
    # Be sure to close the loop.
    polygon = torch.cat((polygon, polygon[..., -1:, :]), axis=-2)

    area = torch.zeros(polygon.shape[:-2], dtype=polygon.dtype, device=polygon.device)
    n = polygon.shape[-2]  # Number of points of the polygon

    rad = torch.deg2rad  # Convert degrees to radians.

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * torch.sin(rad(lat_middle))

    area = area * radius_earth * radius_earth / 2

    return torch.abs(area)

_delta = 0.01  # Reasonable smallest delta in latitude and longitude
_min_patch_area: float = area(
    torch.tensor(
        [
            # The smallest patches will be at the poles. Just use the north pole.
            [90, 0],
            [90, _delta],
            [90 - _delta, _delta],
            [90 - _delta, 0],
        ],
        dtype=torch.float64,
    )
).item()
_area_earth = 4 * np.pi * radius_earth * radius_earth

pos_expansion = FourierExpansion(_delta, 720, assert_range=False)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of latitudes and longitudes in
degrees."""

scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
""":class:`.FourierExpansion`: Fourier expansion for the encoding of patch areas in squared
kilometers."""

absolute_time_expansion = FourierExpansion(1, 367, assert_range=False)
""":class:`.FourierExpansion`: Fourier expansion for the absolute time encoding in days."""

absolute_levels_expansion = FourierExpansion(1, 100)
""":class:`.FourierExpansion`: Fourier expansion for generic z levels"""

pressure_levels_expansion = FourierExpansion(1, 10000)
""":class:`.FourierExpansion`: Fourier expansion for atmospheric pressure levels in hPa"""

depth_levels_expansion = FourierExpansion(1, 10000)
""":class:`.FourierExpansion`: Fourier expansion for ocean depths in meters"""
