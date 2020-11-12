import numpy as np


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