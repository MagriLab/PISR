import torch
from torch import nn

from .layers import (
    TimeDistributedLinear,
    TimeDistributedConv2d,
    TimeDistributedConvTranspose2d,
    TimeDistributedMaxPool2d,
    TimeDistributedUpsamplingBilinear2d
)

from .utils.checks import ValidateDimension


class SuperResolution(nn.Module):

    def __init__(self) -> None:

        super().__init__()

        # TODO >> Need to populate this class.

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

        return x

