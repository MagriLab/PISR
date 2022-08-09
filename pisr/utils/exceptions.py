from typing import Optional

from .enums import eSolverFunction


class DimensionWarning(Warning):

    def __init__(self, message: str) -> None:

        """Warning for dimension-related issues.

        Parameters
        ----------
        message: str
            Warning string to raise.
        """

        super().__init__()
        self.message = message

    def __str__(self) -> str:
        return f'{self.message}'


class DimensionError(Exception):

    def __init__(self,
                 expected: Optional[int] = None,
                 received: Optional[int] = None,
                 msg: Optional[str] = None) -> None:

        """Exception for incompatible dimension issues.

        Parameters
        ----------
        expected: Optional[int]
            Expected number of dimensions.
        received: Optional[int]
            Received number of dimensions.
        msg: Optional[str]
            Message to display.
        """

        super().__init__()

        self.expected = expected
        self.received = received
        self.msg = msg

        if all(v is None for v in [self.expected, self.received, self.msg]):
            raise ValueError('Must pass arguments to exception.')

        if sum(v is not None for v in [self.expected, self.received]) == 1:
            raise ValueError('Must pass both expected and received.')

    def __str__(self) -> str:

        if self.msg:
            return self.msg

        if self.expected == self.received:
            return 'Expected / Received are identical, please check logic...'

        return f'Expected array with ndim={self.expected}, received ndim={self.received}'


class SolverConsistencyError(Exception):

    def __init__(self, solver_type: eSolverFunction) -> None:

        super().__init__()

        self.solver_type = solver_type

    def __str__(self) -> str:

        msg = f'You selected a solver type of: {str(self.solver_type)}, '

        if self.solver_type == eSolverFunction.NONLINEAR:
            msg += 'please ensure C = 0'

        return msg


class SolverConsistencyWarning(Warning):

    def __init__(self, solver_type: eSolverFunction) -> None:

        super().__init__()

        self.solver_type = solver_type

    def __str__(self) -> str:

        msg = f'You selected a solver type of: {str(self.solver_type)}, '

        if self.solver_type == eSolverFunction.NONLINEAR:
            msg += 'please note that C = 0 :: solving diffusion only.'

        return msg
