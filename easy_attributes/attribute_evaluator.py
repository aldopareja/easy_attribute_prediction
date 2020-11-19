from itertools import chain
from pathlib import Path

import numpy as np
from detectron2.config import configurable
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.comm import get_world_size

from easy_attributes.model import OutputHead
from easy_attributes.utils.io import write_serialized


class AttributeEvaluator(DatasetEvaluator):

    @configurable
    def __init__(self, output_head: OutputHead, output_path : Path):
        self.output_head = output_head
        self.output_path = output_path

    @classmethod
    def from_config(cls, cfg):
        return {'output_head': OutputHead(cfg),
                'output_path': Path(cfg.OUTPUT_DIR) / 'last_results.json'}

    def reset(self):
        self._predictions = []
        self._labels = []

    def process(self, inputs, outputs):
        self._labels.append({k:v.cpu()
                             for k,v in inputs['outputs' ].items()})
        self._predictions.append({k:v.cpu()
                                  for k,v in outputs.items()})

    def evaluate(self):

        if get_world_size()>1:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            labels = comm.gather(self._labels, dst=0)

            predictions = list(chain(*predictions))
            labels = list(chain(*labels))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions
            labels = self._labels

        all_preds, all_labels = map(lambda l: {k: np.concatenate([a[k].numpy() for a in l]) for k in l[0].keys()},
                                    [predictions, labels])

        err_dict = self.output_head.pred_error(all_preds, all_labels)

        write_serialized(err_dict, self.output_path)

        return err_dict



