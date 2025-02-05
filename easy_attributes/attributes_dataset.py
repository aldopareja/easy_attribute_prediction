
from itertools import repeat
from pathlib import Path

import numpy as np
from detectron2.config import CfgNode
from detectron2.data import DatasetFromList, MapDataset, DatasetCatalog, MetadataCatalog
import torch
from pycocotools import mask as mask_util

from easy_attributes.utils.io import read_serialized, load_input
from easy_attributes.utils.istarmap_tqdm_patch import array_apply
from easy_attributes.utils.visualize import visualize_data_dict


def data_dict_from_serialized(d, cfg, metadata):
    for input in cfg.DATASETS.INPUTS:
        if metadata['inputs'][input]['type'] == 'bitmask':
            d[input]['counts'] = d[input]['counts'].encode('ascii')
    for output in cfg.DATASETS.OUTPUTS:
        meta_out = metadata['outputs'][output]
        if meta_out['type'] == 'discrete':
            d[output] = meta_out['class_to_index_map'][d[output]]
        elif meta_out['type'] == 'continuous':
            d[output] = np.array([d[output]], dtype=np.float32)
    return d

class CustomMapper():
    def __init__(self, dataset_json_path: Path, cfg: CfgNode):
        self.metadata = read_serialized(dataset_json_path.parent / 'metadata.yml')
        self.cfg = cfg

    def __call__(self, d):
        tensor_input = []
        base_input = load_input(d['filename'])
        tensor_input.append(base_input)
        for input in sorted(self.cfg.DATASETS.INPUTS):
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

        outputs = {k: d[k] for k in self.cfg.DATASETS.OUTPUTS}

        return {'inputs': {'input_tensor': tensor_input},
                'outputs': outputs}

class AttributeDataset():
    def __init__(self, dataset_json_path: Path, cfg):
        self.metadata = read_serialized(dataset_json_path.parent / 'metadata.yml')
        self.cfg = cfg
        data = read_serialized(dataset_json_path)
        data_dicts = array_apply(data_dict_from_serialized, zip(data,
                                                                repeat(cfg),
                                                                repeat(self.metadata)),
                                parallel=True,
                                total=len(data),
                                description='data_dict_from_serialized')
        dataset = DatasetFromList(data_dicts, copy=False)
        mapper = CustomMapper(dataset_json_path, cfg)
        dataset = MapDataset(dataset, mapper)
        self.dataset = dataset

    def get_data_dicts(self):
        return self.dataset

def build_datasets(cfg, data_path: Path):
    metadata = read_serialized(data_path / 'metadata.yml')
    for d in set(cfg.DATASETS.TRAIN + cfg.DATASETS.TEST):
        attr_dataset = AttributeDataset(data_path / (d + '.json'), cfg)
        DatasetCatalog.register(d, lambda ad=attr_dataset: ad.get_data_dicts())
        MetadataCatalog.get(d).set(**metadata)
        MetadataCatalog.get(d).set(thing_classes=['object'])


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
    # loader = DataLoader(dataset,batch_size=2,num_workers=0, shuffle=True)
    # print(iter(loader).next())
