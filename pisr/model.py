from typing import Optional

import torch
from torch import nn

from .layers import TimeDistributedConv2d, TimeDistributedConvTranspose2d


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

        self.activation = nn.Tanh()

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


class SuperResolution(nn.Module):

    def __init__(self, upscaling: int = 3) -> None:

        """Super Resolution model.

        Note :: This is currently hardcoded - needs to be made configurable.

        Parameters:
        ===========
        upscaling: int
            Factor by which to upscale the input tensor.
        """

        super().__init__()

        if upscaling % 2 == 0:
            raise ValueError('Must provide an odd upscaling value...')

        self.upscaling = upscaling

        self.cb1 = ConvBlock(2, 8, 16)
        self.cb2 = ConvBlock(16, 32, 64)

        self.conv_transpose = TimeDistributedConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(self.upscaling, self.upscaling),
            stride=(self.upscaling, self.upscaling),
            padding=0
        )

        self.cb3 = ConvBlock(64, 32, 16)
        self.cb4 = ConvBlock(16, 8, 2)

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


        x = self.cb1(x)
        x = self.cb2(x)

        x = self.conv_transpose(x)

        x = self.cb3(x)
        x = self.cb4(x, last_activation=False)

        return x

