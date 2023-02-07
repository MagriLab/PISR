from typing import Optional, Protocol, TypeVar


T = TypeVar('T')


class Solver(Protocol[T]):

    nk: int
    nk_grid: int

    kk: T
    nabla: T

    ndim: int

    def dynamics(self, u_hat: T) -> T:
        ...

    # TODO >> Should not be part of the Protocol
    def phys_to_fourier(self, t: T) -> T:
        ...

    # TODO >> Should not be part of the Protocol
    def fourier_to_phys(self, t_hat: T, nref: Optional[int]) -> T:
        ...

    # TODO >> Should not be part of the Protocol
    def energy_spectrum(self, u_hat: T, agg: bool) -> T:
        ...
