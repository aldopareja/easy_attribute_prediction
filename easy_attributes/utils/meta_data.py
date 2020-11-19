import numpy as np

from easy_attributes.utils.io import load_input
from easy_attributes.utils.istarmap_tqdm_patch import array_apply


def get_continuous_metadata(data_dicts, attribute_name):
    values = np.array([d[attribute_name] for d in data_dicts])
    std = values.std()
    std = 1.0 if std == 0 else std
    return {'type': 'continuous',
            'average': float(values.mean()),
            'std': float(std)}


def get_discrete_metadata(data_dicts, attribute_name):
    values = [d[attribute_name] for d in data_dicts]
    possible_values = sorted(list(set(values)))
    class_to_index_map = {k: i for i, k in enumerate(possible_values)}

    return {'type': 'discrete',
            'class_to_index_map': class_to_index_map}

def _load_input(d):
    return load_input(d['filename']).numpy()

def get_pixels_mean_and_std(data_dicts):
    all_inputs = array_apply(_load_input,
                             data_dicts,
                             parallel=True,
                             unpack=False)

    all_inputs = np.stack(all_inputs, axis=0)
    return {'pixel_mean': [all_inputs[:,i].mean().item() for i in range(all_inputs.shape[1])],
            'pixel_std': [all_inputs[:,i].std().item() for i in range(all_inputs.shape[1])]}