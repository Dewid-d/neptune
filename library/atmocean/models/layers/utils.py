# ----------------------------------------------------------------------------
# Atmospheric-Ocean coupling Library for sub-seasonal to seasonal forecasting
# Copyright (c) 2023-2024 ***********
# Licensed under The MIT License [see LICENSE for details]
# Written by Davide Donno
# ----------------------------------------------------------------------------

from atmocean.models.layers.fourier import pos_expansion, scale_expansion, absolute_time_expansion
from atmocean.models.layers.posencoding import pos_scale_enc
from atmocean.utils import get_padded_shape
from atmocean.daylength import (
    from_doy_to_date,
    day_length_vector
)

import torch.utils.checkpoint as checkpoint
from itertools import chain, product
from timm.layers import trunc_normal_
from pyperlin import FractalPerlin2D
from functools import lru_cache
from einops import rearrange
from torch import nn
import numpy as np
import torch
import math



def function_to_and_from(x, fn, frm, to, pkwargs: dict, **kwargs):
    x = rearrange(x, f'{frm} -> {to}', **pkwargs)
    x = fn(x, **kwargs)
    x = rearrange(x, f'{to} -> {frm}', **pkwargs)
    return x


def get_activation_fn(act_fn: str):
    # determine activation function
    if act_fn == "relu":
        act_fn = nn.ReLU
    elif act_fn == "gelu":
        act_fn = nn.GELU
    elif act_fn == "silu":
        act_fn = nn.SiLU
    elif act_fn == "sigmoid":
        act_fn = nn.Sigmoid
    else:
        raise ValueError(f"Unknown activation function {act_fn}")
    return act_fn


def init_weights(m: nn.Module):
    """Initialise weights of a module with a truncated normal distribution.

    `nn.LayerNorm` is initialised with a `weight` of 1 and a `bias` of 0.

    Args:
        m (torch.nn.Module): Module.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


def exec_checkpointing(fn, x, active, x2=None):
    if active: 
        if x2 is not None:
            return checkpoint.checkpoint(fn, x, x2, use_reentrant=False)    
        return checkpoint.checkpoint(fn, x, use_reentrant=False)
    else: 
        return fn(x)


def window_partition_2d(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse_2d(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

@lru_cache
def get_attn_mask_2d(input_resolution, win_size, shift_size, sealand_attn_mask=None):
    if any(i > 0 for i in shift_size):
        # calculate attention mask for SW-MSA
        H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
        img_mask = torch.zeros((1, H_, W_, 1))  # 1 H W 1
        h_slices = (slice(0, -win_size[0]),
                    slice(-win_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -win_size[1]),
                    slice(-win_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition_2d(img_mask, win_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, win_size[0] * win_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask + sealand_attn_mask if sealand_attn_mask is not None else attn_mask
    else:
        # attn_mask = None
        attn_mask = sealand_attn_mask if sealand_attn_mask is not None else None
    return attn_mask


def window_partition_3d(x, window_size):
    """
    Args:to_2tuple
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, T, H, W, C = x.shape
    x = x.view(B, T // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
    return windows


def window_reverse_3d(windows, window_size, T, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, T // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    return x

@lru_cache
def get_attn_mask_3d(input_resolution, win_size, shift_size, suppl_attn_mask = None):
    if any(i > 0 for i in shift_size):
        T_, H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
        img_mask = torch.zeros((1, T_, H_, W_, 1))  # 1 H W 1
        cnt = 0
        for d in slice(-win_size[0]), slice(-win_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-win_size[1]), slice(-win_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-win_size[2]), slice(-win_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, win_size)  # nW, ws[0]*ws[1]*ws[2], 1
        mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn_mask = attn_mask + suppl_attn_mask if suppl_attn_mask is not None else attn_mask
    else:
        # attn_mask = None
        attn_mask = suppl_attn_mask if suppl_attn_mask is not None else None
    return attn_mask


def window_partition_4d(x: torch.Tensor, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Partition tokens into their respective windows

     Args:
        x: input tensor (B, D, H, W, T, C)

        window_size: local window size.


    Returns:
        windows: (B*num_windows, window_size*window_size*window_size*window_size, C)
    """
    x_shape = x.size()

    B, T, Z, H, W, C = x_shape
    x = x.view(
        B, 
        T // window_size[0],  # number of windows in time dimension
        window_size[0],  # window size in time dimension
        Z // window_size[1],  # number of windows in deptj dimension
        window_size[1],  # window size in deptj dimension
        H // window_size[2],  # number of windows in height dimension
        window_size[2],  # window size in height dimension
        W // window_size[3],  # number of windows in width dimension
        window_size[3],  # window size in width dimension
        C,
    )
    windows = (x.permute(0, 1, 3, 5, 7, 2, 4, 6, 8, 9)
               .contiguous()
               .view(-1, window_size[0] * window_size[1] * window_size[2] * window_size[3], C))
    return windows


def window_reverse_4d(windows: torch.Tensor, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor (B*num_windows, window_size, window_size, C)
        window_size: local window size.
        dims: dimension values.

    Returns:
        x: (B, D, H, W, T, C)
    """

    B, T, Z, H, W = dims
    x = windows.view( B,
        torch.div(T, window_size[0], rounding_mode="floor"), torch.div(Z, window_size[1], rounding_mode="floor"),
        torch.div(H, window_size[2], rounding_mode="floor"), torch.div(W, window_size[3], rounding_mode="floor"),
        window_size[0], window_size[1], window_size[2], window_size[3],
        -1,
    )
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4, 8, 9).contiguous().view(B, T, Z, H, W, -1)
    return x


@lru_cache
def get_attn_mask_4d(input_resolution, win_size, shift_size, suppl_attn_mask = None):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """
    if any(i > 0 for i in shift_size):
        T_, Z_, H_, W_ = get_padded_shape(shape=input_resolution, patch_size=win_size)
        img_mask = torch.zeros((1, T_, Z_, H_, W_, 1))
        cnt = 0
        for t in slice(-win_size[0]), slice(-win_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for z in slice(-win_size[1]), slice(-win_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for h in slice(-win_size[2]), slice(-win_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    for w in slice(-win_size[3]), slice(-win_size[3], -shift_size[3]), slice(-shift_size[3], None):
                        img_mask[:, t, z, h, w, :] = cnt
                        cnt += 1
        mask_windows = window_partition_4d(img_mask, win_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        # attn_mask = None
        attn_mask = suppl_attn_mask if suppl_attn_mask is not None else None
    return attn_mask


def window_partition_nd(x: torch.Tensor, win_size: tuple[int]):
    x_shape = x.shape
    x = x.view(
        x_shape[0], 
        *tuple(chain(*tuple((xs // ws, ws) for xs, ws in zip(x_shape[1:-1], win_size)))), 
        x_shape[-1],    
    )
    windows = (
        x.permute(
            0, 
            *tuple(
                i for i in (
                    list(range(1, len(win_size)*2+1, 2)) + 
                    list(range(2, len(win_size)*2+1, 2))
                )
            ), 
            len(win_size)*2+1, 
        ).contiguous()
        .view(
            -1, 
            np.prod(win_size), 
            x_shape[-1]
        )
    )
    return windows


def window_reverse_nd(windows: torch.Tensor, win_size: tuple[int], dims):
    B, L, D = dims[0], dims[1:-1], dims[-1]
    window_cuts = tuple([xs // ws for xs, ws in zip(L, win_size)])
    x = windows.view(
        B, 
        *window_cuts, 
        *win_size, 
        D, 
    )
    perm_ids = tuple(chain(*tuple(
        (i, i + len(win_size)) 
        for i in range(1, len(win_size) + 1))))
    last_id = len(win_size)*2 + 1
    x = (
        x.permute(0, *perm_ids, last_id)
        .contiguous()
        .view(B, *L, D)
    )
    return x

@lru_cache
def get_attn_mask_nd(res: tuple[int], win_size: tuple[int], shift_size: tuple[int]):
    if any(i > 0 for i in shift_size):
        pad_res = get_padded_shape(shape = res, patch_size = win_size)
        img_mask = torch.zeros((1, *pad_res, 1))
        cnt = 0
        dim_slices = [
            (slice(-ws), slice(-ws, -ss), slice(-ss, None))
            for ws, ss in zip(win_size, shift_size)
        ]
        for dim_slice in product(*dim_slices):
            img_mask[:, *dim_slice, :] = cnt
            cnt +=1
        mask_windows = window_partition_nd(img_mask, win_size)
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    else:
        attn_mask = None
    return attn_mask


class PerlinNoise(nn.Module):
    def __init__(self, octaves=3, periods=4, factor=0.5):
        super().__init__()
        self.resolutions = [(periods**i,periods**i) for i in range(1,octaves+1)]
        self.factors = [factor**i for i in range(octaves)]
    
    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        for b in range(B):
            for t in range(T):
                generator = torch.Generator(device=x.device)
                noise = FractalPerlin2D(shape=(C,512,1024), resolutions=self.resolutions, factors=self.factors, generator=generator)().to(x.device)[:,:H,:W]
                if torch.is_grad_enabled():
                    x[b,:,t] = x[b,:,t] + noise
        return x



class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
    """
    def __init__(self, sigma=0.01):
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor):
        if torch.is_grad_enabled() and self.sigma != 0:
            noise = (torch.sqrt(torch.as_tensor(self.sigma)) * torch.randn_like(x)).to(x.device)
            x = x + noise
        return x


class DefaultPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        atmos_3d, atmos_2d, ocean_3d, ocean_2d = x
        # B, Ca3d, T, Za3d, H, W = atmos_3d.shape
        # _, Ca2d, _, _, _ = atmos_2d.shape
        # _, Co3d, _, Zo3d, _, _ = ocean_3d.shape
        # _, Co2d, _, _, _ = ocean_2d.shape
        atmos_3d, ocean_3d = (
            rearrange(atmos_3d, 'b c t z h w -> b (c z) t h w'), 
            rearrange(ocean_3d, 'b c t z h w -> b (c z) t h w')
            )
        x = torch.cat((atmos_3d, atmos_2d, ocean_3d, ocean_2d), dim=1)
        return x


class DefaultPostprocess(nn.Module):
    def __init__(self, Ca3d, Za3d, Ca2d, Co3d, Zo3d, Co2d):
        super().__init__()
        self.Ca3d, self.Za3d, self.Ca2d, self.Co3d, self.Zo3d, self.Co2d = Ca3d, Za3d, Ca2d, Co3d, Zo3d, Co2d

    def forward(self, x: torch.Tensor):
        # B, _, T, H, W = x.shape
        A3d = self.Ca3d * self.Za3d
        atmos_3d = rearrange(x[:,0:A3d], 'b (c z) t h w -> b c t z h w', c=self.Ca3d, z=self.Za3d)
        A2d = A3d + self.Ca2d
        atmos_2d = x[:,A3d:A2d]
        O3d = A2d + self.Co3d * self.Zo3d
        ocean_3d = rearrange(x[:,A2d:O3d], 'b (c z) t h w -> b c t z h w', c=self.Co3d, z=self.Zo3d)
        O2d = O3d + self.Co2d
        ocean_2d = x[:,O3d:O2d]
        return atmos_3d, atmos_2d, ocean_3d, ocean_2d


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SpaceTimeConv(nn.Module):
    def __init__(self, in_dim, space_kernel_size, time_kernel_size, temporal_conv=nn.Conv1d, hidden_dim=None, out_dim=None, bias=True):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        self.spatial_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size=space_kernel_size, padding='same', bias=bias)
        self.temporal_conv = temporal_conv(hidden_dim, out_dim, kernel_size=time_kernel_size, bias=bias)

        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x: torch.Tensor):
        b, _, _, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.spatial_conv(x)
        h, w = x.shape[-2:]
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        x = rearrange(x, "b c t h w -> (b h w) c t")
        x = self.temporal_conv(x)
        x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
        return x


class SpaceTimeConv3d(nn.Module):
    def __init__(self, in_dim, space_kernel_size, time_kernel_size, temporal_conv=nn.Conv1d, hidden_dim=None, out_dim=None, bias=True):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim

        self.spatial_conv = nn.Conv3d(in_dim, hidden_dim, kernel_size=space_kernel_size, padding='same', bias=bias)
        self.temporal_conv = temporal_conv(hidden_dim, out_dim, kernel_size=time_kernel_size, bias=bias)

        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)

    def forward(self, x: torch.Tensor):
        b, _, _, z, h, w = x.shape
        x = rearrange(x, "b c t z h w -> (b t) c z h w")
        x = self.spatial_conv(x)
        z, h, w = x.shape[-3:]
        x = rearrange(x, "(b t) c z h w -> b c t z h w", b=b)
        x = rearrange(x, "b c t z h w -> (b z h w) c t")
        x = self.temporal_conv(x)
        x = rearrange(x, "(b z h w) c t -> b c t z h w", h=h, w=w, z=z)
        return x


class PseudoConv4d(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 space_kernel_size: int, 
                 time_kernel_size: int, 
                 space_padding: str = 'same', 
                 temporal_conv = nn.Conv1d, 
                 hidden_dim: int = None, 
                 out_dim: int = None, 
                 bias: bool = True, 
                 channel_last: bool = True, 
            ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        self.spatial_conv = nn.Conv3d(in_dim, hidden_dim, kernel_size=space_kernel_size, padding=space_padding, bias=bias)
        self.temporal_conv = temporal_conv(hidden_dim, out_dim, kernel_size=time_kernel_size, bias=bias)
        nn.init.dirac_(self.temporal_conv.weight.data) # initialized to be identity
        nn.init.zeros_(self.temporal_conv.bias.data)
        self.channel_last = channel_last
        if channel_last: 
            self.pattern_1 = "b t z h w d -> (b t) d z h w"
            self.pattern_2 = "(b z h w) d t -> b t z h w d"
        else: 
            self.pattern_1 = "b d t z h w -> (b t) d z h w"
            self.pattern_2 = "(b z h w) d t -> b d t z h w"
    
    def forward(self, x: torch.Tensor):
        if self.channel_last:
            b, t, z, h, w, d = x.shape
        else:
            b, d, t, z, h, w = x.shape
        x = rearrange(x, self.pattern_1)
        x = self.spatial_conv(x)
        z, h, w = x.shape[-3:]
        # [8, 20, 23, 192, 384]
        x = rearrange(x, "(b t) d z h w -> (b z h w) d t", b=b, t=t)
        x = self.temporal_conv(x)
        x = rearrange(x, self.pattern_2, h=h, w=w, z=z)
        return x


class TimeConv(nn.Module):
    def __init__(self, 
                 dim: int, 
                 out_dim: int=None, 
                 time_steps: int=1):
        super().__init__()
        out_dim = out_dim or dim
        self.dim = dim
        self.out_dim = out_dim
        self.time_steps = time_steps
        self.conv = nn.Conv1d(dim, out_dim, kernel_size=time_steps, stride=time_steps)

    def forward(self, x: torch.Tensor):
        T = self.time_steps
        B, L, _ = x.shape
        # adjust the stride if the passed timesteps are less than the maximum
        x = rearrange(x, "b (t l) c -> (b l) c t", t=T, l=L//T)
        # convolve the input (B, V, T, H, W) to (B, D, 1, H/P, W/P)
        x = self.conv(x)
        # rearrange to the original form
        x = rearrange(x, "(b l) c t -> b t l c", b=B, l=L//T)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeMlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim, bias=True)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim, bias=True)

    def forward(self, t: torch.Tensor):
        t = self.linear1(t)
        t = self.silu(t)
        t = self.linear2(t)
        return t


class TimeMlpShiftScale(nn.Module):
    def __init__(self, in_features: int, out_features: int=None) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.dim = in_features
        self.time_dim = out_features
        self.linear = nn.Linear(in_features, out_features * 2)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.linear(t)
        t = self.silu(t)
        t = rearrange(t, "b d -> b d 1 1 1")
        scale, shift = t.chunk(2, dim=1)
        return x * scale + shift


class ShiftScale(nn.Module):
    def __init__(self, dim, height=None, width=None) -> None:
        super().__init__()
        H, W = height or 1, width or 1
        self.dim = dim
        self.shift = nn.Parameter(torch.zeros(1,dim,1,H,W), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1,dim,1,H,W), requires_grad=True)
    
    def forward(self, x: torch.Tensor):
        return x * self.scale + self.shift


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PositionLinearAttention(nn.Module):
    """Position linear attention"""
    def __init__(self, dim: int, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps
        self.query_conv = nn.Conv1d(dim, dim // 8, kernel_size=1, bias=True)
        self.key_conv = nn.Conv1d(dim, dim // 8, kernel_size=1, bias=True)
        self.value_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=True)
    
    def forward(self, 
                f_a: torch.Tensor, # features from system A
                f_b: torch.Tensor, # features from system B
                **kwargs):
        # Apply the feature map to the queries and keys
        B_, N, D = f_a.shape
        f_a = rearrange(f_a, 'b n d -> b d n')
        f_b = rearrange(f_b, 'b n d -> b d n')
        Q = self.query_conv(f_a).view(B_, -1, N)
        K = self.key_conv(f_b).view(B_, -1, N)
        V = self.value_conv(f_b).view(B_, -1, N)
        
        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)
        
        tailor_sum = 1 / (N + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, D, N)
        
        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)
        
        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(B_, D, N)
        
        f_a = (f_a + self.gamma * weight_value).contiguous()
        f_a = rearrange(f_a, 'b d n -> b n d')
        return f_a


class LatLonTimeEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pos_embed = nn.Linear(dim, dim)
        self.scale_embed = nn.Linear(dim, dim)
        self.absolute_time_embed = nn.Linear(dim, dim)
    
    def forward(self, 
            x: torch.Tensor, 
            t: torch.Tensor, 
        ):
        device = x.device
        _, D, _, H, W = x.shape
        # lat and lon tensors from the image
        lat = torch.as_tensor(np.linspace(-90, 90, int(H * 2))[::-1].copy()).to(device)
        lon = torch.as_tensor(np.linspace(0, 360, int(W * 2))).to(device)
        # get the positional and scale encoding
        pos_encode, scale_encode = pos_scale_enc(
            D, lat, lon, 2, 
            pos_expansion = pos_expansion, 
            scale_expansion = scale_expansion, 
        )
        # get the absolute time encoding
        absolute_time_encode = absolute_time_expansion(t, D)
        # embedding of position, scale and time
        pe = self.pos_embed(pos_encode[None, None, :].to(device))
        se = self.scale_embed(scale_encode[None, None, :].to(device))
        te = self.absolute_time_embed(absolute_time_encode.to(device))[:, :, :, None, None]
        # rearrange the data
        pe = rearrange(pe, f'b t (h w) d -> b d t h w', h=H, w=W) # b d 1 h w
        se = rearrange(se, f'b t (h w) d -> b d t h w', h=H, w=W) # b d 1 h w
        te = rearrange(te, f'b t d h w -> b d t h w')             # b d t 1 1
        # sum to the input
        x = x + pe + se + te
        return x


class SolarRadiationEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.solarrad = nn.Linear(1, dim)
    
    def forward(self, 
            x: torch.Tensor,
            t: torch.Tensor,
        ):
        device = x.device
        _, D, _, H, W = x.shape
        # lat and lon tensors from the image
        lats = torch.as_tensor(np.linspace(-90, 90, int(H))[::-1].copy()).to(device)
        lons = torch.as_tensor(np.linspace(0, 360, int(W))).to(device)
        # get the grid
        doy = t[:,-1].clone()
        date = from_doy_to_date(doy.cpu().numpy())
        # H x W x 1
        day_length = day_length_vector(date, lats.cpu().numpy(), lons.cpu().numpy())
        # to tensor
        day_length = torch.as_tensor(day_length, dtype=torch.float32, device=device)
        # 1 x H x W x B
        day_length = day_length.unsqueeze(0)
        # B x H x W x 1
        day_length = rearrange(day_length, 'd h w b -> b h w d')
        # B x H x W x D
        day_length = self.solarrad(day_length)
        # B x D x H x W
        day_length = rearrange(day_length, 'b h w d -> b d h w')
        # to shape B x D x T x H x W
        day_length = day_length.unsqueeze(2)
        # sum to the input
        x = x + day_length
        return x


# NOTE: adapted from ModAFNO (author: Davide Donno)
class PositionalEmbedding(nn.Module):
    """
    A module for generating positional embeddings based on timesteps.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

        freqs = torch.pi * torch.arange(
            start=1, end=self.num_channels // 2 + 1, dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        x = x.view(-1).outer(self.freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        x = rearrange(x, '(b t) d -> b t d', b=B, t=T)
        return x

# NOTE: adapted from ModAFNO (author: Davide Donno)
class TimeEmbedNet(nn.Module):
    """
    A network that generates a timestep embedding and processes it with an MLP.

    Parameters:
    -----------
    max_time : float, optional
        Maximum input time. The inputs to `forward` is should be in the range [0, max_time].
    dim : int, optional
        The dimensionality of the time embedding.
    depth : int, optional
        The number of layers in the MLP.
    activation_fn:
        The activation function, default GELU.
    method : str, optional
        The embedding method. Either "sinusoidal" (default) or "onehot".
    """

    def __init__(
        self,
        max_time: float = 1.0, 
        dim: int = 64, 
        depth: int = 1, 
        activation_fn = nn.GELU, 
    ):
        super().__init__()
        self.max_time = max_time
        self.sinusoid_embed = PositionalEmbedding(dim)
        self.dim = dim
        blocks = []
        for _ in range(depth):
            blocks.extend([nn.Linear(dim, dim), activation_fn()])
        self.mlp = nn.Sequential(*blocks)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t / self.max_time
        emb = self.sinusoid_embed(t)
        return self.mlp(emb)

# NOTE: adapted from ModAFNO (author: Davide Donno)
class TimeEmbedding(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dimension: int=3) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # time embedding with sinusoid
        self.time_embed = TimeEmbedNet(367, dim=in_dim, depth=3)
        # additive projection for features
        self.time_additive_proj = nn.Linear(in_dim, out_dim)
        self.dimension = dimension
    
    def _forward3d(self, x: torch.Tensor, t: torch.Tensor):
        x = rearrange(x, 'b d z h w -> b z h w d')
        time_emb = self.time_embed(t)
        time_additive = self.time_additive_proj(time_emb)
        time_additive = time_additive.unsqueeze(2).unsqueeze(2)
        # B, T, H, W, D = time_additive.shape
        # res = (T, H, W)
        # time_additive = padding_fn(time_additive, res, [8, 1, 1], False, True)
        x = x + time_additive
        x = rearrange(x, 'b z h w d -> b d z h w')
        return x
    
    def _forward4d(self, x: torch.Tensor, t: torch.Tensor):
        x = rearrange(x, 'b d t z h w -> b t z h w d')
        time_emb = self.time_embed(t)
        time_additive = self.time_additive_proj(time_emb)
        time_additive = time_additive.unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        _, T, Z, H, W, _ = time_additive.shape
        res = (T, Z, H, W)
        # time_additive = padding_fn(time_additive, res, [8, 1, 1, 1], False, True)
        x = x + time_additive
        x = rearrange(x, 'b t z h w d -> b d t z h w')
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if self.dimension == 3:
            return self._forward3d(x, t)
        elif self.dimension == 4:
            return self._forward4d(x, t)
        else:
            raise ValueError(f'Passed number of dimensions is wrong. Can have only dimension = 3, 4. Given {self.dimension}')
