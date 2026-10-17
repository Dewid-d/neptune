from atmocean.models.layers.utils import TimeEmbedNet

from einops import rearrange
from typing import Tuple
import torch.nn as nn
import torch



class AdaptiveTimeLayerNorm(nn.Module):
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
        self.time_emb = TimeEmbedNet(367, context_dim, depth=3)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialise the weights."""
        nn.init.zeros_(self.ln_modulation[-1].weight)
        nn.init.zeros_(self.ln_modulation[-1].bias)

    def forward(self, 
        x: torch.Tensor,        # B x D x T x H x W
        c: torch.Tensor = None, # B x T
    ) -> torch.Tensor:
        B, T, H, W, D = x.shape
        if c is not None:
            c = self.time_emb(c) # B x T x D
            shift, scale = self.ln_modulation(c).chunk(2, dim=-1)
            shift = rearrange(shift, 'b t d -> b t 1 1 d')
            scale = rearrange(scale, 'b t d -> b t 1 1 d')
        else:  
            shift, scale =  0.0, 1.0
        xln = rearrange(x, 'b t h w d -> b (t h w) d')
        xln = self.ln(xln)
        xln = rearrange(xln, 'b (t h w) d -> b t h w d', b=B, d=D, t=T, h=H, w=W)
        x = xln * (self.scale_bias + scale) + shift
        return x



class ChannelLayerNorm(nn.Module):
    """
    Layer Normalization over third-last channel dimension.
    Taken from: https://github.com/ai2cm/ace/blob/main/fme/core/models/conditional_sfno/layers.py#L209
    """
    def __init__(self, 
        dim: int, 
        eps: float = 1e-5, 
        elementwise_affine: bool = False, 
    ):
        super(ChannelLayerNorm, self).__init__()
        self.n_channels = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.constant_(self.weight, 1.0)
            torch.nn.init.constant_(self.bias, 0.0)

    def forward(self, 
            x: torch.Tensor, # input tensor of shape (B, D, T, H, W)
        ) -> torch.Tensor:
        if x.dim() < 3:
            raise ValueError(
                f"Expected at least 4D input with channel at dim=-3, got shape {tuple(x.shape)}"
            )
        if x.size(-4) != self.n_channels:
            raise ValueError(
                f"Channel dimension mismatch: got C={x.size(-4)}, expected {self.n_channels}"
            )

        # Compute per-pixel mean/var across channels without transposing
        mean = x.mean(dim=-4, keepdim=True)
        var = x.var(dim=-4, keepdim=True, unbiased=False)
        inv_std = torch.rsqrt(var + self.eps)
        y = (x - mean) * inv_std

        if self.weight is not None and self.bias is not None:
            # Broadcast [C] over [N, C, *spatial]
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = y * self.weight.view(*shape) + self.bias.view(*shape)
        return y


class ConditionalLayerNorm(nn.Module):
    """
    Conditional Layer Normalization as described in "AdaSpeech: Adaptive
    Text to Speech for Custom Voice" https://arxiv.org/abs/2103.00993.

    Assumes that the input has shape (batch_size, channels, height, width).
    Taken from: https://github.com/ai2cm/ace/blob/main/fme/core/models/conditional_sfno/layers.py#L209
    """
    def __init__(self,
        dim: int, 
        img_shape: Tuple[int, int], 
        dim_doy: int = None, 
        dim_noise: int = None, 
        global_layer_norm: bool = False, 
        epsilon: float = 1e-5, 
        elementwise_affine: bool = True, 
    ):
        super().__init__()
        self.dim = dim
        self.embed_dim_doy = dim_doy or dim
        self.embed_dim_noise = dim_noise or dim
        self.epsilon = epsilon
        if self.embed_dim_doy > 0:
            self.time_embed = TimeEmbedNet(367, self.embed_dim_doy, depth=3)
            self.time_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.embed_dim_doy, dim*2), 
            )
        else:
            self.time_embed = None
            self.time_modulation = None
        if self.embed_dim_noise > 0:
            # no bias as it is already handled in the non-2d layers
            self.W_scale_3d = nn.Conv3d(
                self.embed_dim_noise, self.dim, kernel_size=1, bias=True
            )
            self.W_bias_3d = nn.Conv3d(
                self.embed_dim_noise, self.dim, kernel_size=1, bias=True
            )
        else:
            self.W_scale_3d = None
            self.W_bias_3d = None
        if global_layer_norm:
            self.norm = nn.LayerNorm(
                (self.dim, *img_shape), eps=epsilon, elementwise_affine=elementwise_affine, 
            )
        else:
            self.norm = ChannelLayerNorm(
                self.dim, eps=epsilon, elementwise_affine=elementwise_affine, 
            )
        self._global_layer_norm = global_layer_norm
        self.reset_parameters()

    def reset_parameters(self):
        if self.W_scale_3d is not None:
            nn.init.constant_(self.W_scale_3d.weight, 0.0)
            nn.init.constant_(self.W_scale_3d.bias, 1.0)
        if self.W_bias_3d is not None:
            nn.init.constant_(self.W_bias_3d.weight, 0.0)
            nn.init.constant_(self.W_bias_3d.bias, 1.0)
        # reset parameters of time modulation
        if self.time_modulation is not None:
            nn.init.constant_(self.time_modulation[-1].weight, 0.0)
            nn.init.constant_(self.time_modulation[-1].bias, 0.0)

    def forward_4d(self, x, t = None, c = None):
        BT, D, H, W = x.shape
        # Create scale tensor as all ones
        scale = torch.ones(
            [BT, D, 1, 1], device=x.device, dtype=x.dtype
        )
        # Create bias tensor as all zeros
        bias = torch.zeros(
            [BT, D, 1, 1], device=x.device, dtype=x.dtype
        )

        # Perturb scale and bias according to noise
        if c is not None:
            scale = scale + self.W_scale_3d(c.unsqueeze(2)).squeeze(2)
            bias = bias + self.W_bias_3d(c.unsqueeze(2)).squeeze(2)
        # Perturb scale and bias according to time
        if t is not None:
            t = self.time_embed(t) # B x T x D
            bias_, scale_ = self.time_modulation(t).chunk(2, dim=-1)
            bias_ = rearrange(bias_, 'b t d -> (b t) d 1 1')
            scale_ = rearrange(scale_, 'b t d -> (b t) d 1 1')
            scale = scale + scale_
            bias = bias + bias_
        # Normalize with the conditioning modulation
        if isinstance(self.norm, ChannelLayerNorm):
            xln = self.norm(x.unsqueeze(2)).squeeze(2)
        else:
            xln = rearrange(x, 'bt d h w -> bt (h w) d')
            xln = self.norm(xln)
            xln = rearrange(xln, 'bt (h w) d -> bt d h w', bt=BT, d=D, h=H, w=W)
        return xln * scale + bias

    def forward_5d(self, x, t = None, c = None):
        B, D, T, H, W = x.shape
        # Create scale tensor as all ones
        scale = torch.ones(
            [B, D, T, 1, 1], device=x.device, dtype=x.dtype
        )
        # Create bias tensor as all zeros
        bias = torch.zeros(
            [B, D, T, 1, 1], device=x.device, dtype=x.dtype
        )

        # Perturb scale and bias according to noise
        if c is not None:
            scale = scale + self.W_scale_3d(c)
            bias = bias + self.W_bias_3d(c)

        # Perturb scale and bias according to time
        if t is not None:
            t = self.time_embed(t) # B x T x D
            bias_, scale_ = self.time_modulation(t).chunk(2, dim=-1)
            bias_ = rearrange(bias_, 'b t d -> b d t 1 1')
            scale_ = rearrange(scale_, 'b t d -> b d t 1 1')
            scale = scale + scale_
            bias = bias + bias_

        # Normalize with the conditioning modulation
        if isinstance(self.norm, ChannelLayerNorm):
            xln = self.norm(x)
        else:
            xln = rearrange(x, 'b d t h w -> b (t h w) d')
            xln = self.norm(xln)
            xln = rearrange(xln, 'b (t h w) d -> b d t h w', b=B, d=D, t=T, h=H, w=W)
        return xln * scale + bias

    def forward(self, 
            x: torch.Tensor,        # input tensor of shape (B, D, T, H, W) or (BT, D, H, W)
            t: torch.Tensor = None, # time instant of shape (B, T)
            c: torch.Tensor = None, # conditioning tensor of shape (B, D, T, H, W) or (BT, D, H, W)
        ) -> torch.Tensor:
        """
        Conditional Layer Normalization

        This is a modified version of LayerNorm that allows the scale and bias to be
        conditioned on a context embedding.

        """
        if x.dim() == 5:
            return self.forward_5d(x=x, t=t, c=c)
        elif x.dim() == 4:
            return self.forward_4d(x=x, t=t, c=c)
