import torch
from .utils.types import TypeTensor


def get_low_res_grid(high_res: TypeTensor, factor: int = 3) -> TypeTensor:

    """Produces low-resolution grid from high-resolution grid.

    Note: This will only work with two-dimensional fields.
          Input shape must be (..., N, N).

    Parameters:
    ===========
    high_res: TypeTensor
        High-resolution field to generate low-resolution field from.
    factor: int
        Odd factor by which to downsample the high-resolution field.

    Returns:
    ========
    low_res: TypeTensor
        Low-resolution field sampled from high-resolution field.
    """
    
    if factor % 2 == 0:
        raise ValueError('Must provide an odd factor to allow overlapping sensor measurements.')

    n_high_res = high_res.shape[-1]
    target_res = int(n_high_res / factor)

    if not n_high_res % target_res == 0:
        raise ValueError('High resolution data and factor do not produce valid low resolution field.')

    start_idx = int((factor - 1) / 2)
    
    lr_slice = slice(start_idx, n_high_res, factor)
    low_res = high_res[..., lr_slice, lr_slice]

    if not low_res.shape[-1] == target_res:
        raise ValueError('Unable to produce low resolution field.')

    return low_res

