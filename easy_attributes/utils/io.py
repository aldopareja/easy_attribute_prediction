import json
from pathlib import Path

import numpy as np
import torch
import yaml
import hickle as hkl


def read_serialized(file_path: Path):
    with open(file_path, "r") as f:
        if file_path.name.endswith(".json"):
            return json.load(f)
        elif file_path.name.endswith(".yml"):
            return yaml.full_load(f)
        else:
            raise NotImplementedError


def write_serialized(data, file_path: Path):
    """Write json and yaml file"""
    assert file_path is not None
    if file_path.name.endswith(".json"):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    elif file_path.name.endswith('.yml'):
        with open(file_path, "w") as f:
            yaml.safe_dump(data, f, indent=4)
    else:
        raise ValueError("Unrecognized filename extension", file_path)


def load_input(input_path: Path) -> torch.FloatTensor:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)
    ext = input_path.suffix
    if ext == '.npy':
        input = np.load(input_path)
    elif ext == '.hkl':
        input = hkl.load(str(input_path))
    else:
        raise NotImplementedError

    return torch.FloatTensor(input)
