from typing import Optional

import torch
from torch import nn

from .layers import TimeDistributedConv2d, TimeDistributedConvTranspose2d, TimeDistributedUpsample
from .utils.checks import ValidateDimension


class ConvBlock(nn.Module):

    def __init__(self,
                 input_filters: int,
                 middle_filters: int,
                 output_filters: int,
                 kernel_size: tuple[int, int] = (3, 3)) -> None:

        """Convolutional block with residual connection.

        Parameters:
        ===========
        input_filters: int
            Depth of the input tensor.
        middle_filters: int
            Number of filters to use for the central convolutional layer.
        output_filters: int
            Number of filters to use for the output convolutional layer.
        kernel_size: tuple[int, int]
            Kernel size to use for the convolutional layers.
        """

        super().__init__()

        self.conv1 = TimeDistributedConv2d(
            input_filters,
            middle_filters,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='circular'
        )

        self.conv2 = TimeDistributedConv2d(
            middle_filters,
            middle_filters,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='circular'
        )

        self.conv3 = TimeDistributedConv2d(
            middle_filters,
            output_filters,
            kernel_size=kernel_size,
            padding='same',
            padding_mode='circular'
        )

        self.identity_mapping = self._get_identity_mapping(input_filters, output_filters, kernel_size)

        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _get_identity_mapping(input_filters: int, output_filters: int, kernel_size: tuple[int, int]) -> Optional[nn.Module]:

        """Produce relevant mapping for residual connection.

        Parameters:
        ===========
        input_filters: int
            Depth of the input tensor.
        output_filters: int
            Number of filters to use for the output convolutional layer.
        kernel_size: tuple[int, int]
            Kernel size to use for the convolutional layers.

        Returns:
        ========
        Optional[nn.Module]
            Returns convolutional layer if there is a difference in the input / output filters.

        """

        if input_filters == output_filters:
            return None

        return TimeDistributedConv2d(input_filters, output_filters, kernel_size=kernel_size, padding='same', padding_mode='circular')

    @ValidateDimension(ndim=5)
    def forward(self, x: torch.Tensor, last_activation: bool = True) -> torch.Tensor:

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

        identity = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.identity_mapping is not None:
            identity = self.identity_mapping(x)

        out += identity

        if last_activation:
            out = self.activation(out)

        return out


class SRCNN(nn.Module):

    def __init__(self, lr_nx: int, upscaling: int = 9, mode: str = 'bicubic') -> None:

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
        self.hr_nx = self.lr_nx * self.upscaling

        self._conv_kwargs = dict(padding='same', padding_mode='circular')

        # define layers
        self.activation = nn.ReLU(inplace=True)

        self.upsample = TimeDistributedUpsample((self.hr_nx, self.hr_nx), mode=self.mode)

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


class BlockSRCNN(nn.Module):

    def __init__(self, lr_nx: int, upscaling: int = 9, mode: str = 'bicubic') -> None:

        """BlockSRCNN: SRCNN with additional conv blocks.

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
        self.hr_nx = self.lr_nx * self.upscaling

        self._conv_kwargs = dict(padding='same', padding_mode='circular')

        # define layers
        self.activation = nn.ReLU(inplace=True)

        self.upsample = TimeDistributedUpsample((self.hr_nx, self.hr_nx), mode=self.mode)

        self.conv1 = TimeDistributedConv2d(2, 64, kernel_size=(9, 9), **self._conv_kwargs)
        self.conv2 = TimeDistributedConv2d(64, 32, kernel_size=(5, 5), **self._conv_kwargs)

        self.conv3 = ConvBlock(32, 32, 32)
        self.conv4 = ConvBlock(32, 32, 32)
        self.conv5 = ConvBlock(32, 32, 32)

        self.conv6 = TimeDistributedConv2d(32, 2, kernel_size=(3, 3), **self._conv_kwargs)

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

        x = self.conv3(x, last_activation=True)
        x = self.conv4(x, last_activation=True)
        x = self.conv5(x, last_activation=True)

        x = self.conv6(x)

        return x
