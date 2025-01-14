import torch
from typing import Sequence


def permute_final_dims(tensor: torch.Tensor, indices: Sequence[int]):
    zero_index = -1 * len(indices)
    first_indices = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_indices + [zero_index + i for i in indices])
