import torch
from torch.utils.data import TensorDataset


class UnlabeledTensorDataset(TensorDataset):

    def __init__(self, data_tensor: torch.Tensor) -> None:

        """Dataset for unlabeled tensors.

        Parameters
        ----------
        data_tensor: torch.Tensor
            Tensor to form dataset with.
        """

        super().__init__()
        self.data_tensor = data_tensor

    def __getitem__(self, idx: int) -> torch.Tensor:                                                      # type: ignore
        return self.data_tensor[idx]

    def __len__(self) -> int:
        return len(self.data_tensor)
