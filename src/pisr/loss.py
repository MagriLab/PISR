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

        # analytical derivative
        a_dudt_hat = self.solver.dynamics(u_hat)
        a_dudt_hat = a_dudt_hat[:, :-1, ...]

        # empirical derivative
        e_dudt_hat = (1.0 / self.dt) * (u_hat[:, 1:, ...] - u_hat[:, :-1, ...])

        residual = a_dudt_hat - e_dudt_hat
        residual = oe.contract('ij, btiju -> btiju', self.fwt, residual)

        loss = oe.contract('... -> ', residual * torch.conj(residual)) / residual.numel()

        return loss


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
