import einops
import opt_einsum as oe
import torch

from kolsol.torch.solver import KolSol

from .solver import Solver
from .utils.checks import ValidateDimension


class BaseLoss:

    solver: Solver[torch.Tensor]

    def __init__(self, dt: float, fwt_lb: float) -> None:

        """Base class for the Physics-Informed Loss.

        Parameters
        ----------
        dt: float
            Length of the time-step
        fwt_lb: float
            Fourier weighting term - lower bound.
        """

        self.dt = dt
        self.fwt_lb = fwt_lb

        self.constraints: bool

    @property
    def fwt(self) -> torch.Tensor:

        """Fourier weighting term.

        Returns
        -------
        torch.Tensor
            The Fourier weighting term, given the lower-bound.
        """

        return torch.abs((self.fwt_lb ** -(1.0 / self.solver.nk)) ** -torch.sqrt(self.solver.kk / self.solver.ndim))

    @ValidateDimension(ndim=5)
    def _residual(self, t_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the residual of t_hat in the Fourier domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor, t, in the Fourier domain.

        Returns
        -------
        residual: torch.Tensor
            Residual of tensor, t, in the Fourier domain.
        """

        # analytical derivative
        a_dudt_hat = self.solver.dynamics(t_hat)
        a_dudt_hat = a_dudt_hat[:, :-1, ...]

        # empirical derivative
        e_dudt_hat = (1.0 / self.dt) * (t_hat[:, 1:, ...] - t_hat[:, :-1, ...])

        residual = a_dudt_hat - e_dudt_hat
        residual = oe.contract('ij, btiju -> btiju', self.fwt, residual)

        return residual

    @ValidateDimension(ndim=5)
    def calc_residual_loss(self, u: torch.Tensor) -> torch.Tensor:

        """Calculate the L2 norm of a field, u, in the Fourier domain.

        Parameters
        ----------
        u: torch.Tensor
            Field in the physical domain.

        Returns
        -------
        loss: torch.Tensor
            L2 norm of the field, calculated in the Fourier domain.
        """

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        u_hat = self.solver.phys_to_fourier(u)

        res_u = self._residual(u_hat)
        loss = oe.contract('... -> ', res_u * torch.conj(res_u)) / res_u.numel()

        return loss

    @ValidateDimension(ndim=5)
    def _constraint(self, t_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the constraint on t_hat in the Fourier domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor, t, in the Fourier domain.

        Raises
        ------
        NotImplementedError
            Raises as this method needs to be overwritten before used.
        """

        raise NotImplementedError('BaseLoss:_constraint()')

    @ValidateDimension(ndim=5)
    def calc_constraint_loss(self, u: torch.Tensor) -> torch.Tensor:

        """Calculate the constraint loss for a field, u.

        Parameters
        ----------
        u: torch.Tensor
            Field in the physical domain.

        Raises
        ------
        NotImplementedError
            Raises as this method needs to be overwritten before used.
        """

        raise NotImplementedError('BaseLoss:calc_constraint_loss()')


class KolmogorovLoss(BaseLoss):

    def __init__(self, nk: int, re: float, dt: float, fwt_lb: float, device: torch.device) -> None:

        """Kolmogorov Flow Loss.

        Parameters
        ----------
        nk: int
            Number of symmetric wavenumbers to use.
        re: float
            Reynolds number of the flow.
        dt: float
            Length of the time-step.
        fwt_lb: float
            Fourier weighting term - lower bound.
        device: torch.device
            Device on which to initialise the tensors.
        """

        super().__init__(dt, fwt_lb)
        self.solver = KolSol(nk=nk, nf=4, re=re, ndim=2, device=device)

        self.constraints = True

    def _constraint(self, t_hat: torch.Tensor) -> torch.Tensor:

        """Calculate the continuity on t_hat in the Fourier domain.

        Parameters
        ----------
        t_hat: torch.Tensor
            Tensor, t, in the Fourier domain.

        Returns
        -------
        torch.Tensor
            Constraint of tensor, t, in the Fourier domain.
        """

        return oe.contract('iju, btiju -> bt', self.solver.nabla.to(torch.cfloat), t_hat)

    def calc_constraint_loss(self, u: torch.Tensor) -> torch.Tensor:

        """Calculate continuity loss for physical field.

        Parameters
        ----------
        u: torch.Tensor
            Field to calculate residual loss for.

        Returns
        -------
        torch.Tensor
            Continuity loss of the input field.
        """

        u = einops.rearrange(u, 'b t u i j -> b t i j u')
        u_hat = self.solver.phys_to_fourier(u)

        constraint_u = self._constraint(u_hat)

        # TODO >> Do we want to normalise by constraint_u.numel() or u_hat.numel()
        loss = oe.contract('... -> ', constraint_u * torch.conj(constraint_u)) / constraint_u.numel()

        return loss

