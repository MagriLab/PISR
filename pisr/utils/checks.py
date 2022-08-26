import functools as ft
import itertools as it
from typing import Any, Callable, ParamSpec, TypeVar

from .exceptions import DimensionError, DimensionWarning
from .types import TypeTensor

P = ParamSpec('P')
T = TypeVar('T')


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

            # TODO >> Remove mypy pass once mypy is updated:  https://github.com/python/mypy/pull/13459
            arg_chain = it.chain(args, kwargs.values())                                                   # type: ignore
            if not any(map(lambda _arg: isinstance(_arg, TypeTensor.__args__), arg_chain)):               # type: ignore
                raise DimensionWarning('No arguments with dimensions to check')

            for arg in arg_chain:
                if isinstance(arg, TypeTensor.__args__) and not len(arg.shape) == self.ndim:              # type: ignore
                    raise DimensionError(expected=self.ndim, received=len(arg.shape))

            return fn(*args, **kwargs)

        return _fn
