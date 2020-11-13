import numpy as np
import torch
from detectron2.utils.comm import get_world_size


def get_batch_size(total_batch_size):
    world_size = get_world_size()
    assert (
            total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )

    batch_size = total_batch_size // world_size
    return batch_size


def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.tensor(data, device=device)
    elif isinstance(data, list):
        return [to_device(_, device) for _ in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    else:
        return data
