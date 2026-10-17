# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# Code Adapted from the official Huggingface implementation for ConvNeXT Layers
# ----------------------------------------------------------------------------

import torch
import torch.nn as nn
from einops import rearrange


class ConvNextV2GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1, dim))

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # Compute and normalize global spatial feature maps
        global_features = torch.linalg.vector_norm(hidden_states, ord=2, dim=(1, 2, 3), keepdim=True)
        norm_features = global_features / (global_features.mean(dim=-1, keepdim=True) + 1e-6)
        hidden_states = self.weight * (hidden_states * norm_features) + self.bias + hidden_states
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->ConvNextV2
class ConvNextV2LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNextV2Layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # depthwise conv
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, 
            x: torch.Tensor, 
            res: tuple[int], 
        ) -> torch.Tensor: 
        x1, x2, x3 = res
        pattkw = dict(x1=x1, x2=x2, x3=x3)
        input = x
        x = rearrange(x, 'b (x1 x2 x3) d -> b d x1 x2 x3', **pattkw)
        x = self.dwconv(x)
        x = rearrange(x, 'b d x1 x2 x3 -> b x1 x2 x3 d')
        x = self.layernorm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = rearrange(x, ' b x1 x2 x3 d -> b (x1 x2 x3) d', **pattkw)
        x = input + x
        return x
