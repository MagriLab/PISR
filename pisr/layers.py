from typing import Any, Type

import torch
from torch import nn

from .utils.exceptions import DimensionError


class TimeDistributed(nn.Module):

    def __init__(self, module: nn.Module) -> None:

        """TimeDistributed Layer.

        Allows arbitrary layer operation to be applied over the second dimension.

        For example, apply a layer operation to a Tensor with:
            t.shape -> (batch_dim=1000, time_dim=2, ...)

        The TimeDistributed layer treats the second dimension as another batch.

        Parameters
        ----------
        module: nn.Module
            Module operation to apply over the time-dimension.
        """

        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Conduct a single forward-pass through the wrapped layer.

        Parameters
        ----------
        x: torch.Tensor
            Input Tensor.

        Returns
        -------
        y: torch.Tensor
            Output Tensor.
        """

        if not len(x.shape) > 2:
            raise DimensionError(msg='Input must have more than two dimensions.')

        t, n = x.shape[0], x.shape[1]

        x_reshape = x.contiguous().view(t * n, *x.shape[2:])
        y_reshape = self.module(x_reshape)

        y = y_reshape.contiguous().view(t, n, *y_reshape.shape[1:])

        return y


class TimeDistributedWrapper:

    def __init__(self, module: Type[nn.Module]) -> None:

        """Wrapper to generate TimeDistributed layers.

        Parameters
        ----------
        module: Type[nn.Module]
            Module to make TimeDistributed.
        """

        self.module = module

    def __repr__(self) -> str:
        return f'TimeDistributed({self.module})'

    def __call__(self, *args: Any, **kwargs: Any) -> nn.Module:
        return TimeDistributed(self.module(*args, **kwargs))                                              # type: ignore


# defining TimeDistributed layers
TimeDistributedLinear = TimeDistributedWrapper(nn.Linear)
TimeDistributedConv2d = TimeDistributedWrapper(nn.Conv2d)
TimeDistributedConvTranspose2d = TimeDistributedWrapper(nn.ConvTranspose2d)
TimeDistributedMaxPool2d = TimeDistributedWrapper(nn.MaxPool2d)
TimeDistributedUpsamplingBilinear2d = TimeDistributedWrapper(nn.UpsamplingBilinear2d)

