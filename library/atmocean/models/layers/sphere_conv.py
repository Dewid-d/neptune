import torch.nn.functional as F
from torch import nn
import numpy as np
import torch


class GridGenerator:
    def __init__(self, height: int, width: int, kernel_size, stride=1):
        self.height = height
        self.width = width
        self.kernel_size = kernel_size  # (Kh, Kw)
        self.stride = stride  # (H, W)

    def createSamplingPattern(self):
        """
        :return: (1, H*Kh, W*Kw, (Lat, Lon)) sampling pattern
        """
        kerX, kerY = self.createKernel()  # (Kh, Kw)

        # create some values using in generating lat/lon sampling pattern
        rho = np.sqrt(kerX ** 2 + kerY ** 2)
        Kh, Kw = self.kernel_size
        # when the value of rho at center is zero, some lat values explode to `nan`.
        if Kh % 2 and Kw % 2:
            rho[Kh // 2][Kw // 2] = 1e-8

        nu = np.arctan(rho)
        cos_nu = np.cos(nu)
        sin_nu = np.sin(nu)

        stride_h, stride_w = self.stride
        h_range = np.arange(0, self.height, stride_h)
        w_range = np.arange(0, self.width, stride_w)

        lat_range = ((h_range / self.height) - 0.5) * np.pi
        lon_range = ((w_range / self.width) - 0.5) * (2 * np.pi)

        # generate latitude sampling pattern
        lat = np.array([
            np.arcsin(cos_nu * np.sin(_lat) + kerY * sin_nu * np.cos(_lat) / rho) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lat = np.array([lat for _ in lon_range])  # (W, H, Kh, Kw)
        lat = lat.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # generate longitude sampling pattern
        lon = np.array([
            np.arctan(kerX * sin_nu / (rho * np.cos(_lat) * cos_nu - kerY * np.sin(_lat) * sin_nu)) for _lat in lat_range
        ])  # (H, Kh, Kw)

        lon = np.array([lon + _lon for _lon in lon_range])  # (W, H, Kh, Kw)
        lon = lon.transpose((1, 0, 2, 3))  # (H, W, Kh, Kw)

        # (radian) -> (index of pixel)
        lat = (lat / np.pi + 0.5) * self.height
        lon = ((lon / (2 * np.pi) + 0.5) * self.width) % self.width

        LatLon = np.stack((lat, lon))  # (2, H, W, Kh, Kw) = ((lat, lon), H, W, Kh, Kw)
        LatLon = LatLon.transpose((1, 3, 2, 4, 0))  # (H, Kh, W, Kw, 2) = (H, Kh, W, Kw, (lat, lon))

        H, Kh, W, Kw, d = LatLon.shape
        LatLon = LatLon.reshape((1, H * Kh, W * Kw, d))  # (1, H*Kh, W*Kw, 2)

        return LatLon

    def createKernel(self):
        """
        :return: (Ky, Kx) kernel pattern
        """
        Kh, Kw = self.kernel_size

        delta_lat = np.pi / self.height
        delta_lon = 2 * np.pi / self.width

        range_x = np.arange(-(Kw // 2), Kw // 2 + 1)
        if not Kw % 2:
            range_x = np.delete(range_x, Kw // 2)

        range_y = np.arange(-(Kh // 2), Kh // 2 + 1)
        if not Kh % 2:
            range_y = np.delete(range_y, Kh // 2)

        kerX = np.tan(range_x * delta_lon)
        kerY = np.tan(range_y * delta_lat) / np.cos(range_y * delta_lon)

        return np.meshgrid(kerX, kerY)  # (Kh, Kw)


class SphereConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
        ):
        super(SphereConv2d, self).__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        self.grid_shape = None
        self.grid = None

    def genSamplingPattern(self, h, w):
        gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
        LonLatSamplingPattern = gridGenerator.createSamplingPattern()

        # generate grid to use `F.grid_sample`
        lat_grid = (LonLatSamplingPattern[:, :, :, 0] / h) * 2 - 1
        lon_grid = (LonLatSamplingPattern[:, :, :, 1] / w) * 2 - 1

        grid = np.stack((lon_grid, lat_grid), axis=-1)
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid)
            grid_tensor.requires_grad = False
            # self.register_buffer('grid', grid_tensor)
            self.grid = grid_tensor

    def forward(self, x):
        # Generate Sampling Pattern
        B, C, H, W = x.shape

        if (self.grid_shape is None) or (self.grid_shape != (H, W)):
            self.grid_shape = (H, W)
            self.genSamplingPattern(H, W)

        with torch.no_grad():
            grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
            grid.requires_grad = False

        x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

        # self.weight -> (out_c, in_c, Kh, Kw)
        x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size)

        return x  # (B, out_c, H/stride_h, W/stride_w)
