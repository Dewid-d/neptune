# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno, Gabriele Accarino
# Credits to Gabriele Tramonte for SpectralLoss implementation
# ----------------------------------------------------------------------------

from typing import Optional, List, Any, Tuple
import torch.nn.functional as F
from itertools import product
import torch_harmonics as th
from einops import rearrange
import torch.nn as nn
from torch import amp
from enum import Enum
import numpy as np
import torch
import math

from atmocean.stochastic.crps_makani import (
    _crps_ensemble_kernel, 
    _crps_gauss_kernel, 
    _crps_skillspread_kernel
    )
from atmocean.stochastic.crps import CRPS
from atmocean.utils import log_data


def get_observations_and_forecasts(
        y_atmos_3d_true: torch.Tensor, y_atmos_2d_true: torch.Tensor, 
        y_ocean_3d_true: torch.Tensor, y_ocean_2d_true: torch.Tensor, 
        y_atmos_3d_pred: torch.Tensor, y_atmos_2d_pred: torch.Tensor, 
        y_ocean_3d_pred: torch.Tensor, y_ocean_2d_pred: torch.Tensor):

    y_atmos_true = torch.cat([rearrange(y_atmos_3d_true, "b c t z h w -> b (c z) t h w"), y_atmos_2d_true], dim=1)
    y_atmos_pred = torch.cat([rearrange(y_atmos_3d_pred, "b c t z h w -> b (c z) t h w"), y_atmos_2d_pred], dim=1)
    y_ocean_true = torch.cat([rearrange(y_ocean_3d_true, "b c t z h w -> b (c z) t h w"), y_ocean_2d_true], dim=1)
    y_ocean_pred = torch.cat([rearrange(y_ocean_3d_pred, "b c t z h w -> b (c z) t h w"), y_ocean_2d_pred], dim=1)

    observations = torch.cat([y_atmos_true, y_ocean_true], dim=1)
    forecasts = torch.cat([y_atmos_pred, y_ocean_pred], dim=1)
    return observations, forecasts



def spectral_mse_loss(y_true, y_pred, norm):
    """Spectral MSE implementation. Credits to Gabriele Tramonte, ASC CMCC."""
    # Calcola le trasformate di Fourier delle immagini reali e generate
    real_fft = torch.fft.fftn(y_true, dim=(-2, -1), norm=norm)
    generated_fft = torch.fft.fftn(y_pred, dim=(-2, -1), norm=norm)

    # Calcola i moduli delle trasformate di Fourier
    real_spectrum = torch.abs(real_fft)
    generated_spectrum = torch.abs(generated_fft)

    # Calcola la differenza tra i moduli delle trasformate di Fourier
    spectral_difference = F.mse_loss(real_spectrum, generated_spectrum)
    return spectral_difference



def spectral_l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    """
    Implementation of Spectral MSE Loss on a sphere taken from
    https://github.com/NVIDIA/torch-harmonics/blob/main/notebooks/train_sfno.ipynb
    """
    # compute coefficients
    coeffs = torch.view_as_real(solver.sht(prd - tar))
    coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
    norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
    loss = torch.sum(norm2, dim=(-1,-2))

    if relative:
        tar_coeffs = torch.view_as_real(solver.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_norm2 = torch.sum(tar_norm2, dim=(-1,-2))
        loss = loss / tar_norm2

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


# ATMOREP IMPLEMENTATION OF CRPS LOSS
# CREDITS TO THE LIBRARY
class AtmorepCRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ):
        mu = torch.mean(y_pred, dim = -1)
        std_dev = torch.std(y_pred, dim = -1)
        crps = CRPS(y = y_true, mu = mu, std_dev = std_dev)
        return torch.mean(crps)


class MAEAtmorepCRPSLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.crps_loss = AtmorepCRPSLoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, 
        y_true: torch.Tensor, 
        y_pred: torch.Tensor
    ):
        return self.crps_loss(y_true, y_pred) + self.mae_loss(y_true, torch.mean(y_pred, dim=-1))


# MAKANI IMPLEMENTATION OF SPECTRAL CRPS LOSS
# CREDITS TO THE LIBRARY
class SpectralCRPSLoss(nn.Module):
    def __init__(self, 
                 shape: Tuple[int, int], 
                 grid_type: str = "equiangular", 
                 crps_type: str = "skillspread", # @params=[`cdf`, `skillspread`, `gauss`]
                 absolute: Optional[bool] = True, 
                 alpha: Optional[float] = 1.0, 
                 verbose: Optional[bool] = False, 
            ):
        super().__init__()
        self.shape = shape
        self.grid_type = grid_type
        self.crps_type = crps_type
        self.alpha = alpha
        self.verbose = verbose
        self.sht = th.RealSHT(*shape, grid=grid_type).float()
        
        if (self.crps_type != "skillspread") and (self.alpha < 1.0):
            raise NotImplementedError("The alpha parameter (almost fair CRPS factor) is only supported for the skillspread kernel.")
        
        if crps_type not in ["cdf", "skillspread", "gauss"]:
            raise ValueError(f"Unknown CRPS crps_type {self.crps_type}. Possible values: `cdf`, `skillspread`, `gauss`.")
        
        # if absolute is true, the loss is computed only on the absolute value of the spectral coefficient
        self.absolute = absolute
        
        # get the local l weights
        lmax = self.sht.lmax
        l_weights = torch.ones(lmax).reshape(-1, 1)
        self.register_buffer("l_weights", l_weights, persistent=False)
        
        # get the local m weights
        mmax = self.sht.mmax
        m_weights = 2 * torch.ones(mmax).reshape(1, -1)
        m_weights[0, 0] = 1.0
        self.register_buffer("m_weights", m_weights, persistent=False)
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def forward(self, 
                observations: torch.Tensor,   # B x D x T x H x W
                forecasts: torch.Tensor,      # B x D x T x H x W x E
                spectral_weights: Optional[torch.Tensor] = None, 
            ) -> torch.Tensor:
        if self.verbose: log_data('start', forecasts, observations)
        # get some info from the data
        B, D, T, H, W, E = forecasts.shape
        device, dtype = forecasts.device, forecasts.dtype
        self.sht.to(device)
        forecasts = rearrange(forecasts, 'b d t h w e -> b e d t h w')
        if self.verbose: log_data('rearrange', forecasts, observations)
        # sanity checks
        if forecasts.dim() != 6:
            raise ValueError(f"Error, forecasts tensor expected to have 6 dimensions (B x D x T x H x W x E) but found {forecasts.dim()}.")
        
        # we assume that spatial_weights have NO ensemble dim
        if (spectral_weights is not None) and (spectral_weights.dim() != observations.dim()):
            raise ValueError("the weights have to have the same number of dimensions as observations")
        
        # before anything else compute the transform
        # as the CDF definition doesn't generalize well to more than one-dimensional variables, we treat complex and imaginary part as the same
        with amp.autocast(device_type=self.device_type, enabled=False):
            forecasts = self.sht(forecasts.float()) / 4.0 / math.pi
            observations = self.sht(observations.float()) / 4.0 / math.pi
            if self.verbose: log_data('amp.autocast', forecasts, observations)
        if self.absolute:
            # take the absolute value of the forecasts if the flag is active
            forecasts = torch.abs(forecasts).to(dtype)
            observations = torch.abs(observations).to(dtype)
            if self.verbose: log_data('absolute', forecasts, observations)
        else:
            # otherwise take the real part of the SHT complex transformed data
            # torch.view_as_real() function converts the complex numbers to two
            # real dimensions containing the real and the imaginary part
            # i.e., dims from m1, m2, ..., mi to m1, m2, ..., mi, 2
            forecasts = torch.view_as_real(forecasts).to(dtype)
            observations = torch.view_as_real(observations).to(dtype)
            if self.verbose: log_data('view_as_real', forecasts, observations)
            
            # merge complex dimension after channel dimension and flatten
            # this needs to be undone at the end
            forecasts = torch.movedim(forecasts, 6, 4).flatten(3, 4)
            observations = torch.movedim(observations, 5, 3).flatten(2, 3)
            if self.verbose: log_data('movedim', forecasts, observations)
        
        # we assume the following shapes:
        # forecasts: batch, ensemble, channels, time, mmax, lmax
        # observations: batch, channels, time, mmax, lmax
        B, E, D, T, H, W = forecasts.shape
        
        if spectral_weights is None:
            spectral_weights = self.m_weights * self.l_weights
            spectral_weights = torch.stack([spectral_weights for _ in range(T)], dim=0)
            spectral_weights.to(device)
        else:
            spectral_weights = spectral_weights * self.m_weights * self.l_weights
            spectral_weights = torch.stack([spectral_weights for _ in range(T)], dim=0)
            spectral_weights.to(device)
        if self.verbose: log_data('spectral_weights', spectral_weights)
        
        # if ensemble dim is one dimensional then computing the score is quick:
        if E == 1:
            # in this case, CRPS is straightforward
            crps = torch.abs(observations - forecasts.squeeze(1)).reshape(B, D, T * H * W)
            spectral_weights_split = spectral_weights.reshape(1, 1, T * H * W).to(device)
        else:
            # transpose forecasts: ensemble, batch, channels, time, lat, lon
            forecasts = torch.movedim(forecasts, 1, 0) # E x B x D x T x H x W
            if self.verbose: log_data('movedim', forecasts, observations)
            
            # now we need to transpose the forecasts into ensemble direction.
            # ideally we split spatial dims
            forecasts = forecasts.reshape(E, B, D, T * H * W)
            # observations does not need a transpose, but just a split
            observations = observations.reshape(B, D, T * H * W)
            
            # tile in complex dim, then flatten last 3 dims
            spectral_weights_split = spectral_weights.reshape(1, 1, T * H * W).to(device)
            
            if self.verbose: log_data('reshape+weights', forecasts, observations, spectral_weights_split)
            
            # run appropriate crps kernel to compute it pointwise
            if self.crps_type == "cdf":
                # now, E dimension is local and spatial dim is split further
                # we need to sort the forecasts now
                forecasts, idx = torch.sort(forecasts, dim=0)
                ensemble_weights = torch.ones_like(forecasts, device=device)
                # compute score
                crps = _crps_ensemble_kernel(observations, forecasts, ensemble_weights)
                if self.verbose: log_data('cdf', crps)
            elif self.crps_type == "skillspread":
                ensemble_weights = torch.ones_like(forecasts, device=device)
                # compute score
                crps = _crps_skillspread_kernel(observations, forecasts, ensemble_weights, self.alpha)
                if self.verbose: log_data('skillspread', crps)
            elif self.crps_type == "gauss":
                ensemble_weights = torch.ones_like(forecasts, device=device)
                # compute score
                crps = _crps_gauss_kernel(observations, forecasts, ensemble_weights, self.eps)
                if self.verbose: log_data('gauss', crps)
        
        # perform spatial average of crps score
        crps = torch.sum(crps * spectral_weights_split, dim=-1)
        if self.verbose: log_data('sum', crps)
        
        # finally undo the folding of the complex dimension into the channel dimension
        if not self.absolute:
            crps = crps.reshape(B, -1, 2).sum(dim=-1)
            if self.verbose: log_data('reshape', crps)
        
        # the resulting tensor should have dimension B, C, which is what we return
        # return crps
        return torch.mean(crps)


class EnsMAELoss(nn.Module):
    """
    MAE Loss function applied to the ensemble of forecasts.
    Computes the L1Loss between each ensemble member and the 
    ground truth, averaging the results together.

    """
    def __init__(self):
        super().__init__()

    def forward(self, 
        y_true: torch.Tensor, # B x D x T x H x W
        y_pred: torch.Tensor, # B x D x T x H x W x E
    ) -> torch.Tensor:
        # Change the shape of the data for efficiency
        # from B x D x T x H x W x E to B x E x D x T x H x W
        y_pred = rearrange(y_pred, 'b d t h w e -> b e d t h w')
        # Expand y_true to match y_pred dims
        # `y_true_exp` shape: B x 1 x D x T x H x W
        y_true_exp = y_true.unsqueeze(1)
        # Compute per-pixel absolute errors for all ensemble members
        # `err` shape: B x 1 x D x T x H x W
        err = torch.abs(y_pred - y_true_exp)
        # MAE per ensemble member: reduce space/time dims
        # `per_member_mae` shape: E
        per_member_mae = err.mean(dim=(0,2,3,4,5))
        # Sum (or mean depending on your intended design)
        return per_member_mae.mean()


class EnsMAE_SpectralCRPS_Loss(nn.Module):
    def __init__(self,
            shape: Tuple[int, int],
            grid_type: str = "equiangular",
            crps_type: str = "skillspread", # @params=[`cdf`, `skillspread`, `gauss`]
            absolute: Optional[bool] = True,
            alpha: Optional[float] = 1.0,
            verbose: Optional[bool] = False,
            weights: Optional[float] = [1.0, 1.0],
        ):
        super().__init__()
        self.ens_mae_loss = EnsMAELoss()
        self.spectral_crps_loss = SpectralCRPSLoss(
            shape=shape,
            grid_type=grid_type,
            crps_type=crps_type,
            absolute=absolute,
            alpha=alpha,
            verbose=verbose,
        )
        self.weights = weights

    def forward(self, 
            y_true: torch.Tensor, # B x D x T x H x W
            y_pred: torch.Tensor, # B x D x T x H x W x E
        ):
        weights = self.weights
        return (
            weights[0] * self.ens_mae_loss(y_true, y_pred) + 
            weights[1] * self.spectral_crps_loss(y_true, y_pred)
        )




class LossFunctions(Enum):
    SpectralCRPSLoss = SpectralCRPSLoss
    AtmorepCRPSLoss = AtmorepCRPSLoss
    MAEAtmorepCRPSLoss = MAEAtmorepCRPSLoss
    EnsMAELoss = EnsMAELoss
    L1Loss = nn.L1Loss


class GeneralLoss(nn.Module):
    """
    Implementation of a General Loss function that is constituted 
    by several nested losses. It is useful when dealing with 
    several components of losses.

    `loss_args` argument structure
    ------------------------------
    >> loss_args = {
    >>     'weights': [w1, w2, ..., wM], 
    >>     'cls1': {
    >>         'arg1': 'val1', 
    >>         'arg2': 'val2', 
    >>         ..., 
    >>         'argN': 'valN', 
    >>     }, ..., 'clsM': {...}
    >> }

    """
    def __init__(self, loss_args: dict):
        super().__init__()
        self.loss_args = loss_args
        self.losses = []
        assert 'weights' in loss_args.keys(), 'You have not passed the `weights` to the `loss_args`'
        for cls, args in loss_args.items():
            if cls == 'weights': continue
            loss_fn = LossFunctions[cls].value(**args)
            self.losses.append(loss_fn)
        self.weights = loss_args['weights']
        assert len(self.weights) == len(self.losses), f'Number of weights ({len(self.weights)}) different from number of losses ({len(self.losses)})'

    def forward(self, 
        y_true: torch.Tensor, # B x D x T x H x W
        y_pred: torch.Tensor, # B x D x T x H x W (x E)
        lweight: float = 1.0, # weight assigned to the overall loss
    ):
        loss = torch.tensor(0.0, device=y_true.device)
        loss_dict = dict()
        for loss_fn, weight in zip(self.losses, self.weights): 
            loss_value = weight * loss_fn(y_true, y_pred)
            loss_dict[f'{loss_fn.__class__.__name__}'] = loss_value.item()
            loss = loss + loss_value
        return loss * lweight, loss_dict
