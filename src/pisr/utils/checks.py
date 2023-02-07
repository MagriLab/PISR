import functools as ft
import itertools as it
from typing import Callable, ParamSpec, TypeAlias, TypeVar

import numpy as np
import torch

from .exceptions import DimensionError, DimensionWarning


P = ParamSpec('P')
T = TypeVar('T')

TypeTensor: TypeAlias = np.ndarray | torch.Tensor


class ValidateDimension:

    def __init__(self, ndim: int) -> None:

        """Decorator to validate number of dimensions.

        Parameters
        ----------
        ndim: int
            Expected number of dimensions.
        """

        self.ndim: int = ndim

    def __call__(self, fn: Callable[P, T]) -> Callable[P, T]:

        """Wraps the provided function to check for valid dimensions.

        Parameters
        ----------
        fn: Callable[..., Any]
            Function to wrap / check dimensions.

        Returns
        -------
        _fn: Callable[..., Any]
            Wrapped function.
        """

        @ft.wraps(fn)
        def _fn(*args: P.args, **kwargs: P.kwargs) -> T:

            arg_chain = it.chain(args, kwargs.values())
            if not any(map(lambda _arg: isinstance(_arg, TypeTensor.__args__), arg_chain)):
                raise DimensionWarning('No arguments with dimensions to check')

            for arg in arg_chain:
                if isinstance(arg, TypeTensor.__args__) and not len(arg.shape) == self.ndim:
                    raise DimensionError(expected=self.ndim, received=len(arg.shape))

            return fn(*args, **kwargs)

        return _fn
