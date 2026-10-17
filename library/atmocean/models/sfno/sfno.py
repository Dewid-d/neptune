from atmocean.utils import clean_state_dict_forward_module_from_keys
from atmocean.models.layers.norm import ConditionalLayerNorm
from atmocean.models.layers.utils import (
    LatLonTimeEncoding,
    SolarRadiationEncoding,
)
from atmocean.models.layers.sphere_conv import SphereConv2d
from atmocean.models.neptune import (
    Neptune, 
    NeptuneSystemEmbed3d, 
    NeptuneSystemRecover3d
)

from typing import Tuple, List, Union, Literal
from torch.utils.checkpoint import checkpoint
from functools import partialmethod, partial
import torch.nn.functional as F
import torch_harmonics as th
from einops import rearrange
import torch.nn as nn
import torch

Number = Union[float, int]

# Set warning filter to show each warning only once
import warnings

warnings.filterwarnings("once", category=UserWarning)

from neuralop.layers.normalization_layers import AdaIN, InstanceNorm, BatchNorm
from neuralop.layers.embeddings import GridEmbeddingND, GridEmbedding2D
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.complex import CGELU, ctanh, ComplexValued
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.utils import validate_scaling_factor
from neuralop.layers.padding import DomainPadding
from neuralop.models.base_model import BaseModel



class FourierNeuralOperatorBlocks(nn.Module):
    """FNOBlocks implements a sequence of Fourier layers.
    
    The Fourier layers are first described in [1]_, and the exact implementation details 
    of the Fourier layer architecture are discussed in [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels to Fourier layers
    out_channels : int
        Number of output channels after Fourier layers
    n_modes : int or List[int]
        Number of modes to keep along each dimension in frequency space.
        Can either be specified as an int (for all dimensions) or an iterable
        with one number per dimension
    resolution_scaling_factor : Optional[Union[Number, List[Number]]], optional
        Factor by which to scale outputs for super-resolution, by default None
    n_layers : int, optional
        Number of Fourier layers to apply in sequence, by default 1
    max_n_modes : int or List[int], optional
        Maximum number of modes to keep along each dimension, by default None
    fno_block_precision : str, optional
        Floating point precision to use for computations. Options: "full", "half", "mixed", by default "full"
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default True
    channel_mlp_dropout : float, optional
        Dropout parameter for self.channel_mlp, by default 0
    channel_mlp_expansion : float, optional
        Expansion parameter for self.channel_mlp, by default 0.5
    non_linearity : torch.nn.F module, optional
        Nonlinear activation function to use between layers, by default F.gelu
    stabilizer : Literal["tanh"], optional
        Stabilizing module to use between certain layers. Options: "tanh", None, by default None
    norm : Literal["ada_in", "group_norm", "instance_norm", "batch_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", "batch_norm", None, by default None
    ada_in_features : int, optional
        Number of features for adaptive instance norm above, by default None
    preactivation : bool, optional
        Whether to call forward pass with pre-activation, by default False
        If True, call nonlinear activation and norm before Fourier convolution
        If False, call activation and norms after Fourier convolutions
    fno_skip : str, optional
        Module to use for FNO skip connections. Options: "linear", "soft-gating", "identity", None, by default "linear"
        If None, no skip connection is added. See layers.skip_connections for more details
    channel_mlp_skip : str, optional
        Module to use for ChannelMLP skip connections. Options: "linear", "soft-gating", "identity", None, by default "soft-gating"
        If None, no skip connection is added. See layers.skip_connections for more details

    Other Parameters
    -------------------
    complex_data : bool, optional
        Whether the FNO's data takes on complex values in space, by default False
    separable : bool, optional
        Separable parameter for SpectralConv, by default False
    factorization : str, optional
        Factorization parameter for SpectralConv. Options: "tucker", "cp", "tt", None, by default None
    rank : float, optional
        Rank parameter for SpectralConv, by default 1.0
    conv_module : BaseConv, optional
        Module to use for convolutions in FNO block, by default SpectralConv
    joint_factorization : bool, optional
        Whether to factorize all spectralConv weights as one tensor, by default False
    fixed_rank_modes : bool, optional
        Fixed_rank_modes parameter for SpectralConv, by default False
    implementation : str, optional
        Implementation parameter for SpectralConv. Options: "factorized", "reconstructed", by default "factorized"
    decomposition_kwargs : dict, optional
        Kwargs for tensor decomposition in SpectralConv, by default dict()

    References
    -----------
    .. [1] Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
           Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.
    .. [2] Kossaifi, J., Kovachki, N., Azizzadenesheli, K., Anandkumar, A. "Multi-Grid
           Tensorized Fourier Neural Operator for High-Resolution PDEs" (2024).
           TMLR 2024, https://openreview.net/pdf?id=AWiDlO63bH.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        img_shape=[181,360],
        resolution_scaling_factor=None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_channel_mlp=True,
        channel_mlp_dropout=0,
        channel_mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        channel_mlp_skip="soft-gating",
        complex_data=False,
        separable=False,
        factorization=None,
        rank=1.0,
        conv_module=SpectralConv,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        grid: str = "equiangular",
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.resolution_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(resolution_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.channel_mlp_skip = channel_mlp_skip
        self.complex_data = complex_data

        self.use_channel_mlp = use_channel_mlp
        self.channel_mlp_expansion = channel_mlp_expansion
        self.channel_mlp_dropout = channel_mlp_dropout
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.img_shape = img_shape

        # apply real nonlin if data is real, otherwise CGELU
        if self.complex_data:
            self.non_linearity = CGELU
        else:
            self.non_linearity = non_linearity

        self.convs = nn.ModuleList(
            [
                conv_module(
                    self.in_channels,
                    self.out_channels,
                    self.n_modes,
                    resolution_scaling_factor=None
                    if resolution_scaling_factor is None
                    else self.resolution_scaling_factor[i],
                    max_n_modes=max_n_modes,
                    rank=rank,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    fno_block_precision=fno_block_precision,
                    decomposition_kwargs=decomposition_kwargs,
                    complex_data=complex_data,
                    sht_grids=grid,
                )
                for i in range(n_layers)
            ]
        )

        if fno_skip is not None and fno_skip != "none":
            self.fno_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.fno_skips = None
        if self.complex_data and self.fno_skips is not None:
            self.fno_skips = nn.ModuleList([ComplexValued(x) for x in self.fno_skips])

        if self.use_channel_mlp:
            self.channel_mlp = nn.ModuleList(
                [
                    ChannelMLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * channel_mlp_expansion),
                        dropout=channel_mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            if self.complex_data:
                self.channel_mlp = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp]
                )
            if channel_mlp_skip is not None and channel_mlp_skip != "none":
                self.channel_mlp_skips = nn.ModuleList(
                    [
                        skip_connection(
                            self.in_channels,
                            self.out_channels,
                            skip_type=channel_mlp_skip,
                            n_dim=self.n_dim,
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.channel_mlp_skips = None
            if self.complex_data and self.channel_mlp_skips is not None:
                self.channel_mlp_skips = nn.ModuleList(
                    [ComplexValued(x) for x in self.channel_mlp_skips]
                )

        # Each block will have 2 norms if we also use a ChannelMLP
        self.n_norms = 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [InstanceNorm() for _ in range(n_layers * self.n_norms)]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )

        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    BatchNorm(n_dim=self.n_dim, num_features=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )

        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "cond_ln":
            self.norm = nn.ModuleList(
                [
                    ConditionalLayerNorm(
                        out_channels, img_shape=img_shape, global_layer_norm=False
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, batch_norm, ada_in]"
            )

        if self.complex_data and self.norm is not None:
            self.norm = nn.ModuleList([ComplexValued(x) for x in self.norm])

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if self.norm is not None:
            if len(embeddings) == 1:
                for norm in self.norm:
                    norm.set_embedding(embeddings[0])
            else:
                for norm, embedding in zip(self.norm, embeddings):
                    norm.set_embedding(embedding)

    def forward(self, 
            x:   torch.Tensor,        # input tensor
            c:   torch.Tensor = None, # conditioning
            doy: torch.Tensor = None, # day of year
            index=0,                  # index of FNO block
            output_shape=None,        # output shape
        ):
        # add perturbation if it is not provided, to make the model stochastic
        # if c is None:
        #     c = torch.randn_like(x, dtype=x.dtype, device=x.device)
        if self.preactivation:
            return self.forward_with_preactivation(
                x=x, c=c, doy=doy, index=index, output_shape=output_shape
            )
        else:
            return self.forward_with_postactivation(
                x=x, c=c, doy=doy, index=index, output_shape=output_shape
            )

    def forward_with_postactivation(self, 
            x:   torch.Tensor,        # input tensor
            c:   torch.Tensor = None, # conditioning
            doy: torch.Tensor = None, # day of year
            index=0,                  # index of FNO block
            output_shape=None,        # output shape
        ):
        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        if self.norm is not None:
            norm = self.norm[self.n_norms * index]
            if isinstance(norm, ConditionalLayerNorm):
                x_fno = norm(x_fno, t=doy, c=c)
            else:
                x_fno = norm(x_fno)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        if self.norm is not None:
            norm = self.norm[self.n_norms * index + 1]
            if isinstance(norm, ConditionalLayerNorm):
                x = norm(x, t=doy, c=c)
            else:
                x = norm(x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, 
            x:   torch.Tensor,        # input tensor
            c:   torch.Tensor = None, # conditioning
            doy: torch.Tensor = None, # day of year
            index=0,                  # index of FNO block
            output_shape=None,        # output shape
        ):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            norm = self.norm[self.n_norms * index]
            if isinstance(norm, ConditionalLayerNorm):
                x_fno = norm(x_fno, t=doy, c=c)
            else:
                x_fno = norm(x_fno)

        if self.fno_skips is not None:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.use_channel_mlp and self.channel_mlp_skips is not None:
            x_skip_channel_mlp = self.channel_mlp_skips[index](x)
            x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)

        x = x_fno + x_skip_fno if self.fno_skips is not None else x_fno

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        if self.norm is not None:
            norm = self.norm[self.n_norms * index + 1]
            if isinstance(norm, ConditionalLayerNorm):
                x = norm(x, t=doy, c=c)
            else:
                x = norm(x)

        if self.use_channel_mlp:
            if self.channel_mlp_skips is not None:
                x = self.channel_mlp[index](x) + x_skip_channel_mlp
            else:
                x = self.channel_mlp[index](x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        for i in range(self.n_layers):
            self.convs[i].n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)


# customized version of the model
class FourierNeuralOperatorNet(BaseModel, name="FourierNeuralOperator"):
    """N-Dimensional Fourier Neural Operator. The FNO learns a mapping between
    spaces of functions discretized over regular grids using Fourier convolutions,
    as described in [1]_.

    The key component of an FNO is its SpectralConv layer (see
    ``neuralop.layers.spectral_convolution``), which is similar to a standard CNN
    conv layer but operates in the frequency domain.

    For a deeper dive into the FNO architecture, refer to :ref:`fno_intro`.

    Main Parameters
    ---------------
    n_modes : Tuple[int, ...]
        Number of modes to keep in Fourier Layer, along each dimension.
        The dimensionality of the FNO is inferred from len(n_modes).
        n_modes must be larger enough but smaller than max_resolution//2 (Nyquist frequency)
    in_channels : int
        Number of channels in input function. Determined by the problem.
    out_channels : int
        Number of channels in output function. Determined by the problem.
    hidden_channels : int
        Width of the FNO (i.e. number of channels).
        This significantly affects the number of parameters of the FNO.
        Good starting point can be 64, and then increased if more expressivity is needed.
        Update lifting_channel_ratio and projection_channel_ratio accordingly since they are proportional to hidden_channels.
    n_layers : int, optional
        Number of Fourier Layers. Default: 4

    Other parameters
    ---------------
    lifting_channel_ratio : Number, optional
        Ratio of lifting channels to hidden_channels.
        The number of lifting channels in the lifting block of the FNO is
        lifting_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    projection_channel_ratio : Number, optional
        Ratio of projection channels to hidden_channels.
        The number of projection channels in the projection block of the FNO is
        projection_channel_ratio * hidden_channels (e.g. default 2 * hidden_channels).
    positional_embedding : Union[str, nn.Module], optional
        Positional embedding to apply to last channels of raw input
        before being passed through the FNO.
        Options:
        - "grid": Appends a grid positional embedding with default settings to the last channels of raw input.
          Assumes the inputs are discretized over a grid with entry [0,0,...] at the origin and side lengths of 1.
        - GridEmbeddingND: Uses this module directly (see :mod:`neuralop.embeddings.GridEmbeddingND` for details).
        - GridEmbedding2D: Uses this module directly for 2D cases.
        - None: Does nothing.
        Default: "grid"
    non_linearity : nn.Module, optional
        Non-Linear activation function module to use. Default: F.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"], optional
        Normalization layer to use. Options: "ada_in", "group_norm", "instance_norm", None. Default: None
    complex_data : bool, optional
        Whether the data is complex-valued. If True, initializes complex-valued modules. Default: False
    use_channel_mlp : bool, optional
        Whether to use an MLP layer after each FNO block. Default: True
    channel_mlp_dropout : float, optional
        Dropout parameter for ChannelMLP in FNO Block. Default: 0
    channel_mlp_expansion : float, optional
        Expansion parameter for ChannelMLP in FNO Block. Default: 0.5
    channel_mlp_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in channel-mixing mlp. Options: "linear", "identity", "soft-gating", None.
        Default: "soft-gating"
    fno_skip : Literal["linear", "identity", "soft-gating", None], optional
        Type of skip connection to use in FNO layers. Options: "linear", "identity", "soft-gating", None.
        Default: "linear"
    resolution_scaling_factor : Union[Number, List[Number]], optional
        Layer-wise factor by which to scale the domain resolution of function.
        Options:
        - None: No scaling
        - Single number n: Scales resolution by n at each layer
        - List of numbers [n_0, n_1,...]: Scales layer i's resolution by n_i
        Default: None
    domain_padding : Union[Number, List[Number]], optional
        Percentage of padding to use.
        Options:
        - None: No padding
        - Single number: Percentage of padding to use along all dimensions
        - List of numbers [p1, p2, ..., pN]: Percentage of padding along each dimension
        Default: None
    fno_block_precision : str, optional
        Precision mode in which to perform spectral convolution.
        Options: "full", "half", "mixed". Default: "full". Default: "full"
    stabilizer : str, optional
        Whether to use a stabilizer in FNO block. Options: "tanh", None. Default: None.
        stabilizer greatly improves performance in the case `fno_block_precision='mixed'`.
    max_n_modes : Tuple[int, ...], optional
        Maximum number of modes to use in Fourier domain during training.
        None means that all the n_modes are used.
        Tuple of integers: Incrementally increase the number of modes during training.
        This can be updated dynamically during training.
    factorization : str, optional
        Tensor factorization of the FNO layer weights to use.
        Options: "None", "Tucker", "CP", "TT"
        Other factorization methods supported by tltorch. Default: None
    rank : float, optional
        Tensor rank to use in factorization. Default: 1.0
        Set to float <1.0 when using TFNO (i.e. when factorization is not None).
        A TFNO with rank 0.1 has roughly 10% of the parameters of a dense FNO.
    fixed_rank_modes : bool, optional
        Whether to not factorize certain modes. Default: False
    implementation : str, optional
        Implementation method for factorized tensors.
        Options: "factorized", "reconstructed". Default: "factorized"
    decomposition_kwargs : dict, optional
        Extra kwargs for tensor decomposition (see `tltorch.FactorizedTensor`). Default: {}
    separable : bool, optional
        Whether to use a separable spectral convolution. Default: False
    preactivation : bool, optional
        Whether to compute FNO forward pass with resnet-style preactivation. Default: False
    conv_module : nn.Module, optional
        Module to use for FNOBlock's convolutions. Default: SpectralConv

    Examples
    ---------

    >>> from neuralop.models import FNO
    >>> model = FNO(n_modes=(12,12), in_channels=1, out_channels=1, hidden_channels=64)
    >>> model
    FNO(
    (positional_embedding): GridEmbeddingND()
    (fno_blocks): FNOBlocks(
        (convs): SpectralConv(
        (weight): ModuleList(
            (0-3): 4 x DenseTensor(shape=torch.Size([64, 64, 12, 7]), rank=None)
        )
        )
            ... torch.nn.Module printout truncated ...

    References
    -----------
    .. [1] :

    Li, Z. et al. "Fourier Neural Operator for Parametric Partial Differential
        Equations" (2021). ICLR 2021, https://arxiv.org/pdf/2010.08895.

    """
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        img_shape: Tuple[int,int]=[181,360],
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
        grid: str = "equiangular",
    ):
        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__()
        self.n_dim = len(n_modes)

        # n_modes is a special property - see the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers

        # init lifting and projection channels using ratios w.r.t hidden channels
        self.lifting_channel_ratio = lifting_channel_ratio
        self.lifting_channels = int(lifting_channel_ratio * self.hidden_channels)

        self.projection_channel_ratio = projection_channel_ratio
        self.projection_channels = int(projection_channel_ratio * self.hidden_channels)

        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.channel_mlp_skip = (channel_mlp_skip,)
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.complex_data = complex_data
        self.fno_block_precision = fno_block_precision

        ## Positional embedding
        if positional_embedding == "grid":
            spatial_grid_boundaries = [[0.0, 1.0]] * self.n_dim
            self.positional_embedding = GridEmbeddingND(
                in_channels=self.in_channels,
                dim=self.n_dim,
                grid_boundaries=spatial_grid_boundaries,
            )
        elif isinstance(positional_embedding, GridEmbedding2D):
            if self.n_dim == 2:
                self.positional_embedding = positional_embedding
            else:
                raise ValueError(
                    f"Error: expected {self.n_dim}-d positional embeddings, got {positional_embedding}"
                )
        elif isinstance(positional_embedding, GridEmbeddingND):
            self.positional_embedding = positional_embedding
        elif positional_embedding is None:
            self.positional_embedding = None
        else:
            raise ValueError(
                f"Error: tried to instantiate FNO positional embedding with {positional_embedding},\
                              expected one of 'grid', GridEmbeddingND"
            )

        ## Domain padding
        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                resolution_scaling_factor=resolution_scaling_factor,
            )
        else:
            self.domain_padding = None

        ## Resolution scaling factor
        if resolution_scaling_factor is not None:
            if isinstance(resolution_scaling_factor, (float, int)):
                resolution_scaling_factor = [resolution_scaling_factor] * self.n_layers
        self.resolution_scaling_factor = resolution_scaling_factor

        ## FNO blocks
        self.fno_blocks = FourierNeuralOperatorBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            img_shape=img_shape,
            resolution_scaling_factor=resolution_scaling_factor,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=channel_mlp_dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=channel_mlp_skip,
            complex_data=complex_data,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            conv_module=conv_module,
            n_layers=n_layers,
            grid=grid, 
        )

        ## Lifting layer
        # if adding a positional embedding, add those channels to lifting
        lifting_in_channels = self.in_channels
        if self.positional_embedding is not None:
            lifting_in_channels += self.n_dim
        # if lifting_channels is passed, make lifting a Channel-Mixing MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = ChannelMLP(
                in_channels=lifting_in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels,
                n_layers=1,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
            )
        # Convert lifting to a complex ChannelMLP if self.complex_data==True
        if self.complex_data:
            self.lifting = ComplexValued(self.lifting)

        ## Projection layer
        self.projection = ChannelMLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity,
        )
        if self.complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, 
            x: torch.Tensor, 
            c: torch.Tensor = None, 
            doy: torch.Tensor = None, 
            output_shape=None, 
            **kwargs, 
        ):
        """FNO's forward pass

        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity)

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor

        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.

            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        if kwargs:
            warnings.warn(
                f"FNO.forward() received unexpected keyword arguments: {list(kwargs.keys())}. "
                "These arguments will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(
                x=x, c=c, doy=doy, index=layer_idx, output_shape=output_shape[layer_idx]
            )

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class EncoderDecoder(nn.Module):
    def __init__(self,
        img_shape: tuple,
        in_channels: int,
        hidden_channels: int=None,
        out_channels: int=None,
        dims: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation_fn: str = F.gelu,
        dropout_prob: float = 0.0,
        conv_local_module: str = "default", # ['default', 'spherical']
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.dims = dims
        if conv_local_module == 'default':
            if dims == 2:
                conv_module = partial(nn.Conv2d, kernel_size=kernel_size, stride=stride, padding=padding)
            elif dims == 3:
                conv_module = partial(nn.Conv3d, kernel_size=kernel_size, stride=stride, padding=padding)
        elif conv_local_module == 'sphere-conv':
            if dims != 2: raise ValueError(f'If conv_local_module is "spherical", dims must be equal to 2. Passed dimensions {dims}')
            conv_module = partial(SphereConv2d, kernel_size=kernel_size, stride=stride, padding=padding)
        elif conv_local_module == 'disco':
            if dims != 2: raise ValueError(f'If conv_local_module is "spherical", dims must be equal to 2. Passed dimensions {dims}')
            conv_module = partial(th.DiscreteContinuousConvS2, in_shape=tuple(img_shape), out_shape=tuple(img_shape), kernel_shape=kernel_size)
        else:
            raise ValueError(f'You passed the wrong conv_local_module. Expected one of ["default", "spherical"], given {conv_local_module}.')
        self.conv1 = conv_module(in_channels, hidden_channels)
        self.norm1 = ConditionalLayerNorm(hidden_channels, img_shape=img_shape)
        self.conv2 = conv_module(hidden_channels, out_channels)
        self.norm2 = ConditionalLayerNorm(out_channels, img_shape=img_shape)
        self.act = activation_fn
        self.drop = nn.Dropout(dropout_prob)
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self,
        x: torch.Tensor,
        c: torch.Tensor = None,
        t: torch.Tensor = None,
    ):
        if self.dims == 3:
            B, D, T, H, W = x.shape
            patt_in = 'b d t h w -> b d (t h w)'
            patt_out = 'b d (t h w) -> b d t h w'
            patt_kw = dict(b=B, t=T, h=H, w=W)
        elif self.dims == 2:
            BT, D, H, W = x.shape
            patt_in = 'b d h w -> b d (h w)'
            patt_out = 'b d (h w) -> b d h w'
            patt_kw = dict(b=BT, h=H, w=W)
        # apply projection to skip conn
        res = rearrange(x, patt_in, **patt_kw)
        res = self.proj(res)
        patt_kw['d'] = res.shape[1]
        res = rearrange(res, patt_out, **patt_kw)
        # first conv block
        x = self.conv1(x)
        x = self.norm1(x=x, c=c, t=t)
        x = self.act(x)
        x = self.drop(x)
        # second conv block
        x = self.conv2(x)
        x = self.norm2(x=x, c=c, t=t)
        x = self.act(x)
        x = self.drop(x)
        # add skip and return
        return x + res



class NeptuneOceaNO(Neptune):
    def __init__(self, 
        drivers: dict,  # dictionary of drivers
        dim: int,       # input dimension
        insteps: int,
        outsteps: int,
        # NO params
        n_modes: Tuple[int, ...],
        in_img_shape: Tuple[int,int] = None,
        img_shape: Tuple[int,int] = [181,360],
        kernel_size:int = 3,
        stride:int = 1,
        padding:int = 0,
        n_encdec_layers: int = 4,
        n_fno_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: str = "gelu",
        norm: Literal["ada_in", "group_norm", "instance_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        big_skip: Literal["linear", "identity", "soft-gating", None] = None,
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: str = "spectral",
        conv_local_module: str = "default", # @['default', 'spherical']
        ensemble_size: int = None,
        use_checkpointing: bool = False,
        solar_radiation: bool = False,
        sht_grid: str = "equiangular",
    ) -> None:
        super().__init__()
        self.insteps = insteps
        self.outsteps = outsteps
        self.ensemble_size = ensemble_size if ensemble_size is not None else 1
        self.dim = dim
        self.use_checkpointing = use_checkpointing
        self.solar_radiation = solar_radiation

        if non_linearity == "gelu": non_linearity = F.gelu
        elif non_linearity == "silu": non_linearity = F.silu
        elif non_linearity == "relu": non_linearity = F.relu
        else: raise ValueError(f"non_linearity must be one of [gelu, silu, relu]. Given {non_linearity}")

        self.conv_module = conv_module
        if conv_module == "spectral":
            conv_module = SpectralConv
            dims = 3
            self.data_padding = (1,1,32,32)
        elif conv_module == "spherical":
            conv_module = SphericalConv
            dims = 2
            self.data_padding = (1,1,1,1)
        else: raise ValueError(f"conv_module must be one of [spectral, spherical]. Given {conv_module}")

        self.phy_to_latent = NeptuneSystemEmbed3d(
            drivers=drivers, dim=dim, kernel_size=3, 
        )
        self.pos_enc = LatLonTimeEncoding(dim)
        if solar_radiation:
            self.solarrad_enc = SolarRadiationEncoding(dim)

        in_img_shape = in_img_shape or img_shape
        if in_img_shape[0] != img_shape[0] and in_img_shape[1] != img_shape[1]:
            self.need_sampling = True
            self.downsample = th.ResampleS2(
                nlat_in=in_img_shape[0],
                nlon_in=in_img_shape[1],
                nlat_out=img_shape[0],
                nlon_out=img_shape[1],
                grid_in=sht_grid,
                grid_out=sht_grid,
                mode="bilinear",
            )
            self.upsample = th.ResampleS2(
                nlat_in=img_shape[0],
                nlon_in=img_shape[1],
                nlat_out=in_img_shape[0],
                nlon_out=in_img_shape[1],
                grid_in=sht_grid,
                grid_out=sht_grid,
                mode="bilinear",
            )
        else:
            self.need_sampling = False

        self.encoder = nn.ModuleList([
            EncoderDecoder(
                img_shape=img_shape,
                in_channels=dim,
                hidden_channels=dim,
                out_channels=dim,
                dims=dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation_fn=non_linearity,
                dropout_prob=dropout,
                conv_local_module=conv_local_module,
            )
            for _ in range(n_encdec_layers)
        ])

        self.fourier_neural_operator = FourierNeuralOperatorNet(
            n_modes=n_modes,
            img_shape=img_shape,
            in_channels=dim,
            out_channels=dim,
            hidden_channels=dim,
            n_layers=n_fno_layers,
            lifting_channel_ratio=lifting_channel_ratio,
            projection_channel_ratio=projection_channel_ratio,
            positional_embedding=positional_embedding,
            non_linearity=non_linearity,
            norm=norm,
            complex_data=complex_data,
            use_channel_mlp=use_channel_mlp,
            channel_mlp_dropout=dropout,
            channel_mlp_expansion=channel_mlp_expansion,
            channel_mlp_skip=channel_mlp_skip,
            fno_skip=fno_skip,
            resolution_scaling_factor=resolution_scaling_factor,
            domain_padding=domain_padding,
            fno_block_precision=fno_block_precision,
            stabilizer=stabilizer,
            max_n_modes=max_n_modes,
            factorization=factorization,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            separable=separable,
            preactivation=preactivation,
            conv_module=conv_module,
            grid=sht_grid,
        )

        self.decoder = nn.ModuleList([
            EncoderDecoder(
                img_shape=img_shape,
                in_channels=2*dim,
                hidden_channels=dim,
                out_channels=dim,
                dims=dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation_fn=non_linearity,
                dropout_prob=dropout,
                conv_local_module=conv_local_module,
            )
            for _ in range(n_encdec_layers)
        ])

        if big_skip is not None:
            self.big_skip = skip_connection(
                dim, dim, skip_type=big_skip, n_dim=dims,
            )
        else: self.big_skip = None

        self.latent_to_phy = NeptuneSystemRecover3d(
            drivers=drivers, dim=dim, kernel_size=3, insteps=insteps, outsteps=outsteps, 
        )

    def forward(self, 
            x: torch.Tensor,           # input data
            t: torch.Tensor = None,    # day of year
            c: torch.Tensor = None,    # conditioning (B x D x T x H x W)
            drivers: dict = {},        # I/O drivers
            agg_systems: bool = False, # aggregate systems after the recovery
            **kwargs,                  # other keyword arguments (for compatibility among the architectures)
        ) -> torch.Tensor:

        # preprocess inputs
        x, rs = self.preprocess(x, self.data_padding)
        (a3d, a2d, o3d, o2d), _ = self.deaggregate_systems(x, drivers)

        # pre-conv to embed all the systems
        x = self.phy_to_latent((a3d, a2d, o3d, o2d), drivers, False)

        if self.need_sampling:
            x = self.downsample(x)

        # spatial and temporal encoding
        x = self.pos_enc(x, t)

        # solar radiation encoding
        if self.solar_radiation:
            x = self.solarrad_enc(x, t)

        # define initial condition x0 of ensemble
        B, _, T, _, _ = x.shape
        if self.conv_module == "spherical":
            x = rearrange(x, "b d t h w -> (b t) d h w")
        elif self.conv_module == "spectral":
            x = x

        # skip connection forward
        if self.big_skip: skip = self.big_skip(x)

        # create random noise for conditioning
        if c is None:
            c = torch.randn_like(x, dtype=x.dtype, device=x.device)

        enc_features = []
        # convolutional encoder module
        for enc in self.encoder:
            if self.use_checkpointing:
                x = checkpoint(enc, x=x, c=c, t=t, use_reentrant=False)
            else:
                x = enc(x=x, c=c, t=t)
            enc_features.insert(0, x)

        # backbone forward for each ensemble member
        if self.use_checkpointing:
            x = checkpoint(self.fourier_neural_operator, x=x, c=c, doy=t, use_reentrant=False)
        else:
            x = self.fourier_neural_operator(x=x, c=c, doy=t)

        # convolutional decoder module
        for dec, enc_feat in zip(self.decoder, enc_features):
            x = torch.cat((enc_feat, x), dim=1)
            if self.use_checkpointing:
                x = checkpoint(dec, x=x, c=c, t=t, use_reentrant=False)
            else:
                x = dec(x=x, c=c, t=t)

        # add skip connection
        if self.big_skip: x = x + skip

        # post-process data only if we have spherical convolution
        if self.conv_module == "spherical":
            x = rearrange(x, "(b t) d h w -> b d t h w", b=B, t=T)

        if self.need_sampling:
            x = self.upsample(x)

        # recover for each ensemble member
        x = [self.latent_to_phy(x, drivers)]

        # postprocess outputs
        x = [self.postprocess(xi, rs, self.data_padding) for xi in x]

        # stack ensemble members
        x = torch.stack(x, dim=-1) if agg_systems else [torch.stack(xi, dim = -1) if xi[0] is not None else None for xi in list(map(list, zip(*x)))]

        # if ensemble size is only 1 remove dim
        if self.ensemble_size == 1:
            x = [xi[..., 0] if xi is not None else None for xi in x]

        return x

    def load_state_dict(self, state_dict, strict = True, assign = False):
        print(f'Using the custom implementation of load_state_dict()')
        default_load_state_dict = super().load_state_dict
        # clean the state dictionary from the `_forward_module` key within the keys
        state_dict = clean_state_dict_forward_module_from_keys(state_dict)
        # Load model weights
        mkeys, ukeys = default_load_state_dict(state_dict, strict=False)
        print(f'State dict keys: {state_dict.keys()}')
        print(f'Missing keys: {mkeys}')
        print(f'Unexpected keys: {ukeys}')
        # remove `_metadata` key as it doesn't contain anything
        if '_metadata' in state_dict.keys():
            print(f'    _metadata key: {state_dict["_metadata"]}')
            del state_dict["_metadata"]
        # If we have missing keys
        if len(mkeys) > 0:
            print(f'We have missing keys')
            # Check if all the keys belong to recovery layers 
            assert all(['recover' in key for key in mkeys]), f'Missing keys not recognized. Got {mkeys}'
            print(f'All the missing keys belong to the recovery layers')
            print(f'hence, we can assume that we have an ensemble of 3 members')
            # If the assertion goes without errors, then we 
            # can assume that we have an ensemble of 3 members
            # print(f'We have these keys: {state_dict.keys()}')
            ensemble_size = 3
            rec_dict = {
                key:value 
                for key,value in state_dict.items() if 'recover' in key
            }
            new_rec_dict = {}
            for i in range(1, ensemble_size):
                for key,value in rec_dict.items():
                    # create the new key of the state dictionary
                    new_key = key.replace('.0.', f'.{i}.')
                    # the weight will be affected by a small perturbation
                    new_weight = value.clone() + torch.rand_like(value) * 1e-4
                    # update the dictionary of recovery weights
                    new_rec_dict.update({new_key:new_weight})
            state_dict = {**state_dict, **new_rec_dict}
            print(f'Created an updated version of the checkpoint state with these new keys {new_rec_dict.keys()}')
        # Load the updated model weights
        return default_load_state_dict(state_dict, strict=strict, assign=assign)
