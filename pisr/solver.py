from typing import Optional, Protocol, TypeVar

import numpy as np
import torch

T = TypeVar('T', np.ndarray, torch.Tensor)


class Solver(Protocol[T]):

    nk: int
    kk: T
    nabla: T

    ndim: int

    def dynamics(self, u_hat: T) -> T:
        """dynamics"""

    def phys_to_fourier(self, t: T) -> T:
        """phys_to_fourier"""

    def fourier_to_phys(self, t_hat: T, nref: Optional[int]) -> T:
        """fourier_to_phys"""
