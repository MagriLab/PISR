from typing import TypeVar, Union

import numpy as np
import torch

TypeTensor = Union[np.ndarray, torch.Tensor]
T = TypeVar('T', np.ndarray, torch.Tensor)

