from typing import TypeVar

import numpy as np
import torch


T = TypeVar('T', np.ndarray, torch.Tensor)


def get_low_res_grid(high_res: T, factor: int = 3) -> T:

    """Produces low-resolution grid from high-resolution grid.

    Note: This will only work with two-dimensional fields.
          Input shape must be (..., N, N).

    Parameters:
    -----------
    high_res: np.ndarray | torch.Tensor
        High-resolution field to generate low-resolution field from.
    factor: int
        Odd factor by which to downsample the high-resolution field.

    Returns:
    --------
    low_res: np.ndarray | torch.Tensor
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
