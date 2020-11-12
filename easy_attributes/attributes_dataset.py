
from itertools import repeat
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pycocotools import mask as mask_util
from yacs.config import CfgNode

from easy_attributes.utils.io import read_serialized, load_input
from easy_attributes.utils.istarmap_tqdm_patch import array_apply


def data_dict_from_serialized(d, cfg, metadata):
    for input in cfg.DATASET.INPUTS:
        if metadata['inputs'][input]['type'] == 'bitmask':
            d[input]['counts'] = d[input]['counts'].encode('ascii')
    for output in cfg.DATASET.OUTPUTS:
        meta_out = metadata['outputs'][output]
        if meta_out['type'] == 'discrete':
            d[output] = meta_out['class_to_index_map'][d[output]]
        elif meta_out['type'] == 'continuous':
            d[output] = np.float32(d[output])
    return d


class AttributeDataset(Dataset):
    def __init__(self, dataset_json_path: Path, cfg):
        self.metadata = read_serialized(dataset_json_path.parent / 'metadata.yml')
        self.cfg = cfg
        data = read_serialized(dataset_json_path)
        self.data = array_apply(data_dict_from_serialized, zip(data, repeat(cfg), repeat(self.metadata)),
                                parallel=True,
                                total=len(data),
                                description='data_dict_from_serialized')

    def __getitem__(self, idx):
        d = self.data[idx]
        tensor_input = []
        base_input = load_input(d['filename'])
        tensor_input.append(base_input)
        for input in sorted(self.cfg.DATASET.INPUTS):
            in_type = self.metadata['inputs'][input]['type']
            if in_type == 'input_tensor':
                continue
            elif in_type == 'bounding_box':
                box = d[input]
                segmented = np.zeros_like(base_input.numpy(), dtype=np.float32)
                segmented[:, box[1]:box[3] + 1, box[0]:box[2] + 1] = \
                    base_input[:, box[1]:box[3] + 1, box[0]:box[2] + 1]
                tensor_input.append(torch.FloatTensor(segmented))
            elif in_type == 'bitmask':
                mask = d[input]
                mask = mask_util.decode(mask)
                segmented = base_input * torch.FloatTensor(mask)
                tensor_input.append(segmented)
            else:
                raise NotImplementedError

        tensor_input = torch.cat(tensor_input, dim=0)

        outputs = {k: d[k] for k in self.cfg.DATASET.OUTPUTS}

        return {'inputs': {'input_tensor': tensor_input},
                'outputs': outputs}

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    data_path = Path(parser.parse_args().data_path)
    metadata = read_serialized(data_path / 'metadata.yml')
    cfg = CfgNode()
    cfg.DATASET = CfgNode()
    cfg.DATASET.INPUTS = ('file_name', 'mask', 'bbox')
    cfg.DATASET.OUTPUTS = tuple(metadata['outputs'].keys())

    dataset = AttributeDataset(data_path / 'val.json', cfg)
    loader = DataLoader(dataset,batch_size=2,num_workers=0, shuffle=True)
    print(iter(loader).next())
