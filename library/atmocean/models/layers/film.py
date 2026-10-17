# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Updated by Davide Donno to use explicit time dimension T
# ----------------------------------------------------------------------------

"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

`AdaptiveLayerNorm` was inspired by the following file:

    https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101
"""
from einops import rearrange
import torch.nn as nn
import torch

class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalisation with scale and shift modulation."""

    def __init__(self, dim: int, context_dim: int, scale_bias: float = 0) -> None:
        """Initialise.

        Args:
            dim (int): Input dimension.
            context_dim (int): Dimension of the conditioning signal.
            scale_bias (float, optional): Scale bias to add to the scaling factor. Defaults to `0`.
        """
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(context_dim, dim * 2))
        self.scale_bias = scale_bias

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights."""
        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape `(B, T, L, D)`.
            c (torch.Tensor): Conditioning tensor of shape `(B, T, D)`.

        Returns:
            torch.Tensor: Output tensor of shape `(B, T, L, D)`.
        """
        _, L, _ = x.shape
        if c is not None:
            T = c.shape[1]
            x = rearrange(x, "B (T L) C -> B T L C", T=T, L=L//T)
            shift, scale = self.ln_modulation(c).unsqueeze(2).chunk(2, dim=-1)
        else:  shift, scale =  0.0, 1.0
        x = self.ln(x) * (self.scale_bias + scale) + shift
        if c is not None:
            x = rearrange(x, "B T L C -> B (T L) C")
        return x