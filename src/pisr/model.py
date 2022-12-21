import torch
from torch import nn

from .layers import TimeDistributedConv2d, TimeDistributedPeriodicUpsample
from .utils.checks import ValidateDimension


class SRCNN(nn.Module):

    def __init__(self, lr_nx: int, upscaling: int = 15, mode: str = 'bicubic') -> None:

        """SRCNN: Standard CNN for Super-Resolution.

        Parameters:
        ===========
        lr_nx: int
            Low-resolution grid-points.
        upscaling: int
            Upsampling factor for the network.
        mode: str
            Mode of upsampling for the first layer of the network.
        """

        super().__init__()

        if upscaling % 2 == 0:
            raise ValueError('Must provide an odd upscaling value...')

        if mode not in ['bilinear', 'bicubic']:
            raise ValueError('Choose relevant mode for upsampling...')

        self.upscaling = upscaling
        self.mode = mode

        self.lr_nx = lr_nx

        self._conv_kwargs = dict(padding='same', padding_mode='circular')

        # define layers
        self.activation = nn.ReLU(inplace=True)

        self.upsample = TimeDistributedPeriodicUpsample(mode=self.mode, scale_factor=self.upscaling, npad=self.lr_nx // 2)

        self.conv1 = TimeDistributedConv2d(2, 64, kernel_size=(9, 9), **self._conv_kwargs)
        self.conv2 = TimeDistributedConv2d(64, 32, kernel_size=(5, 5), **self._conv_kwargs)
        self.conv3 = TimeDistributedConv2d(32, 2, kernel_size=(5, 5), **self._conv_kwargs)

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Forward pass through the model.

        Parameters:
        ===========
        x: torch.Tensor
            Tensor to pass through the model.

        Returns:
        ========
        torch.Tensor
            Output of the model.
        """

        x = self.upsample(x)

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = self.conv3(x)

        return x


class VDSRBlock(nn.Module):

    def __init__(self, n_filters: int, kernel_size: int) -> None:

        super().__init__()

        self.n_filters = n_filters
        self.kernel_size = kernel_size

        _conv_dict = dict(
                in_channels=self.n_filters,
                out_channels=self.n_filters,
                kernel_size=self.kernel_size,
                padding='same',
                padding_mode='circular'
        )

        self.activation = nn.ReLU(inplace=True)
        self.conv = TimeDistributedConv2d(**_conv_dict)

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor, activation: bool = True) -> torch.Tensor:

        """Forward pass through the model.

        Parameters:
        ===========
        x: torch.Tensor
            Tensor to pass through the model.
        activation: bool
            Whether to use the activation function.

        Returns:
        ========
        torch.Tensor
            Output of the model.
        """

        if activation:
            return self.activation(self.conv(x))

        return self.conv(x)


class VDSR(nn.Module):

    n_layers: int = 16

    n_filters: int = 64
    kernel_size: int = 3

    def __init__(self, lr_nx: int, upscaling: int = 9, mode: str = 'bicubic') -> None:

        """VDSR: VGG-Net based architecture.

        Parameters:
        ===========
        lr_nx: int
            Low-resolution grid-points.
        upscaling: int
            Upsampling factor for the network.
        mode: str
            Mode of upsampling for the first layer of the network.
        """

        super().__init__()

        if upscaling % 2 == 0:
            raise ValueError('Must provide an odd upscaling value...')

        if mode not in ['bilinear', 'bicubic']:
            raise ValueError('Choose relevant mode for upsampling...')

        self.upscaling = upscaling
        self.mode = mode

        self.lr_nx = lr_nx

        _conv_dict = dict(
            in_channels=self.n_filters,
            out_channels=self.n_filters,
            kernel_size=self.kernel_size,
            padding='same',
            padding_mode='circular'
        )

        # define activation
        self.activation = nn.ReLU(inplace=True)

        # define layers
        self.upsample = TimeDistributedPeriodicUpsample(mode=self.mode, scale_factor=self.upscaling, npad=self.lr_nx // 2)
        self.vdsr_layers = nn.ModuleList([VDSRBlock(self.n_filters, self.kernel_size) for _ in range(self.n_layers)])
        self.final_layer = VDSRBlock(self.n_filters, self.kernel_size)

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Forward pass through the model.

        Parameters:
        ===========
        x: torch.Tensor
            Tensor to pass through the model.

        Returns:
        ========
        torch.Tensor
            Output of the model.
        """

        x = self.upsample(x)
        residual = x

        for layer in self.vdsr_layers:
            x = layer(x, activation=True)

        x = self.final_layer(x, activation=False) + residual

        return x
