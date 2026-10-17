# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------- ACKNOWLEDGEMENT --------------------------------
# @Author: Davide Donno
# Code adapted from makani library and re-used for Atlas Global Ocean Emulator
# ---------------------------------------------------------------------------------

import numpy as np
import torch



def rankdata(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    ordinal ranking along dimension dim
    """
    ndim = x.dim()
    perm = torch.argsort(x, dim=dim, descending=False, stable=True)
    
    idx = torch.arange(x.shape[dim], device=x.device).reshape([-1 if i == dim else 1 for i in range(ndim)])
    rank = torch.empty_like(x, dtype=torch.long).scatter_(dim=dim, index=perm, src=idx.expand_as(perm)) + 1
    return rank


def _crps_ensemble_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    CRPS ensemble score from integrating the PDF piecewise
    compare https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py#L7
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism
    """
    
    # beware: forecasts are assumed sorted in sorted order
    # get nanmask
    nanmasks = torch.logical_or(torch.isnan(forecasts), torch.isnan(weights))
    
    # compute total weights
    nweights = torch.where(nanmasks, 0.0, weights)
    total_weight = torch.sum(nweights, dim=0)
    
    # initial values
    obs_cdf = torch.zeros_like(observation)
    forecast_cdf = torch.zeros_like(observation)
    prev_forecast = torch.zeros_like(observation)
    integral = torch.zeros_like(observation)
    nanmask = torch.zeros_like(observation, dtype=torch.bool)
    
    # split lists
    nanmasklist = torch.split(nanmasks, 1, dim=0)
    weightslist = torch.split(weights, 1, dim=0)
    forecastlist = torch.split(forecasts, 1, dim=0)
    for n, token in enumerate(zip(forecastlist, weightslist, nanmasklist)):
        
        # extract variables
        tmpforecast, weight, tmpnanmask = token
        
        # update nanmask
        nanmask = torch.logical_or(tmpnanmask, nanmask)
        
        forecast = torch.where(tmpnanmask, prev_forecast, tmpforecast)
        
        # compute condition
        condition = torch.logical_and(observation < forecast, torch.abs(obs_cdf) < 1.0e-7)
        
        # compute terms
        term_true = (observation - prev_forecast) * torch.square(forecast_cdf) + (forecast - observation) * torch.square(forecast_cdf - 1)
        term_false = (forecast - prev_forecast) * torch.square(forecast_cdf - obs_cdf)
        increment = torch.where(condition, term_true, term_false)
        
        # compute integral
        integral = integral + torch.where(nanmask, 0.0, increment)
        
        # update cdf
        # this only gets updated for values which are not nan
        obs_cdf_new = torch.where(condition, 1.0, obs_cdf)
        obs_cdf = torch.where(nanmask, obs_cdf, obs_cdf_new)
        forecast_cdf = forecast_cdf + weight / total_weight
        
        # update forcast
        prev_forecast = forecast
    
    integral = integral + torch.where(torch.abs(obs_cdf) < 1.0e-7, observation - forecast, 0.0)
    
    # set to nan for first forecasts nan
    integral = torch.where(nanmasklist[0], torch.nan, integral)
    
    return torch.squeeze(integral, dim=0)


def _crps_skillspread_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    alternative CRPS variant that uses spread and skill
    """
    
    observation = observation.unsqueeze(0)
    
    # get nanmask
    nanmasks = torch.logical_or(torch.isnan(forecasts), torch.isnan(weights))
    nanmask = torch.sum(nanmasks, dim=0).bool()
    
    # compute total weights
    nweights = torch.where(nanmasks, 0.0, weights)
    total_weight = torch.sum(nweights, dim=0, keepdim=True)
    
    # get the ranks for the spread computation
    rank = rankdata(forecasts, dim=0)
    
    #  ensemble size
    num_ensemble = forecasts.shape[0]
    
    # get the ensemble spread (total_weight is ensemble size here)
    espread = 2 * torch.mean((2 * rank - num_ensemble - 1) * forecasts, dim=0) * (float(num_ensemble) - 1.0 + alpha) / float(num_ensemble * (num_ensemble - 1))
    eskill = (observation - forecasts).abs().mean(dim=0)
    
    # crps = torch.where(nanmasks.sum(dim=0) != 0, torch.nan, eskill - 0.5 * espread)
    crps = eskill - 0.5 * espread
    
    # set to nan for first forecasts nan
    crps = torch.where(nanmask, torch.nan, crps)
    
    return crps


def _crps_gauss_kernel(observation: torch.Tensor, forecasts: torch.Tensor, weights: torch.Tensor, eps: float) -> torch.Tensor:
    """
    CRPS Gauss score, assuming the input ensemble is gaussian distributed
    disabling torch compile for the moment due to very long startup times when training large ensembles with ensemble parallelism
    """
    
    # compute mean var over observations
    mu = torch.mean(forecasts * weights, dim=0)
    sigma = torch.sqrt(torch.mean(torch.square(forecasts - mu.unsqueeze(0)) * weights, dim=0))
    
    # protect against too small standard deviations
    sigma = torch.clamp(sigma, min=eps)
    
    # compute normalized observation
    obs_norm = (observation - mu) / sigma
    
    # compute normalization
    sqrtpi_inv = 1.0 / np.sqrt(np.pi)
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    
    # compute PDF and CDF
    pdf = sqrtpi_inv * sqrt2_inv * torch.exp(-0.5 * torch.square(obs_norm))
    cdf2m1 = torch.erf(obs_norm * sqrt2_inv)
    
    # compute score
    crps = sigma * (obs_norm * cdf2m1 + 2.0 * pdf - sqrtpi_inv)
    
    return crps
