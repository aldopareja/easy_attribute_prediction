from math import ceil
from typing import Tuple, List, Dict

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, Backbone,  build_backbone
from detectron2.structures import ImageList
from numpy import cumsum
from torch import nn

from easy_attributes.utils.trainer_utils import get_batch_size, to_device


class OutputHead:

    @configurable
    def __init__(self,
                 output_names: Tuple[str],
                 output_metadata: Dict[str, Dict]):
            self.terms = output_names

            loss_funcs = {'continuous': nn.MSELoss,
                          'discrete': nn.CrossEntropyLoss}
            self.loss_methods = {k: loss_funcs[output_metadata[k]['type']]()
                                 for k in self.terms}

            self.terms_indices = self.get_termn_indices(output_names, output_metadata)

            self.out_metadata = output_metadata

    @classmethod
    def from_config(cls,cfg):
        return {'output_names': cfg.DATASETS.OUTPUTS,
                'output_metadata':MetadataCatalog.get(cfg.DATASETS.TEST[0]).outputs}

    def get_termn_indices(self, terms, output_metadata):
        lengths = []
        for term in terms:
            t_meta = output_metadata[term]
            if t_meta['type'] == 'continuous':
                lengths.append(1)
            elif t_meta['type'] == 'discrete':
                lengths.append(len(t_meta['class_to_index_map']))
            else:
                raise NotImplementedError

        lengths = cumsum([0] + lengths)
        self._len = lengths[-1]

        term_indices = {k: list(range(lengths[i], lengths[i+1]))
                        for i,k in enumerate(terms)}

        return term_indices

    def batch_to_terms_dicts(self,input_batch):
        out = {}
        for term in self.terms:
            out[term] = input_batch[:, self.terms_indices[term]]
            meta = self.out_metadata[term]
            if meta['type'] == 'continuous':
                #unstandardize to let predictions be interpretable
                out[term] = out[term] * meta['std'] + meta['average']
        return out

    def loss(self, inputs, targets):
        loss_dict = {}
        for term, loss_method in self.loss_methods.items():

            #unstandardize to keep training stable
            normalize = self.out_metadata[term]['type'] == 'continuous'
            l_input, l_target = map(lambda z: z/self.out_metadata[term]['std'] if normalize
                                              else z,
                                    [inputs[term],targets[term]])

            loss = loss_method(l_input, l_target)

            loss_dict[term] = loss

        # loss_dict["total_loss"] = sum(loss_dict.values())
        return loss_dict

    def __len__(self):
        return self._len


@META_ARCH_REGISTRY.register()
class CustomModel(nn.Module):

    @configurable
    def __init__(self,
                 *,
                 backbone: Backbone,
                 output_head: OutputHead,
                 pixel_mean: Tuple[float],
                 pixel_std: Tuple[float],
                 input_height: int,
                 input_width: int,
                 # batch_size: int,
                 resnet_features: List[str],
                 last_hid_num_feats: int,
                 ):

        super().__init__()
        self.backbone = backbone
        self.backbone_features = resnet_features

        feat_strides = {p: backbone.output_shape()[p].stride for p in resnet_features}
        feat_heights = {p: ceil(input_height / fs) for p, fs in feat_strides.items()}
        feat_widths = {p: ceil(input_width / fs) for p, fs in feat_strides.items()}

        self.poolers = {p: nn.MaxPool2d((feat_heights[p],feat_widths[p])) for p in resnet_features}

        self.sum_feat_channels = sum([backbone.output_shape()[p].channels for p in resnet_features])

        self.output_network = nn.Sequential(nn.Linear(self.sum_feat_channels, last_hid_num_feats),
                                            nn.ReLU(),
                                            nn.Linear(last_hid_num_feats, len(output_head)))

        self.output_head = output_head

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"


    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {'backbone': backbone,
                'output_head': OutputHead(cfg),
                "pixel_mean": cfg.MODEL.PIXEL_MEAN,
                "pixel_std": cfg.MODEL.PIXEL_STD,
                "input_height": MetadataCatalog.get(cfg.DATASETS.TEST[0]).inputs['file_name']['height'],
                "input_width": MetadataCatalog.get(cfg.DATASETS.TEST[0]).inputs['file_name']['width'],
                # "batch_size": get_batch_size(cfg.SOLVER.IMS_PER_BATCH),
                "resnet_features": cfg.MODEL.FPN_OUT_FEATS,
                "last_hid_num_feats": cfg.MODEL.LAST_HIDDEN_LAYER_FEATS
                }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        inputs = self.preprocess_input([batched_inputs['inputs']['input_tensor']])

        features = self.backbone(inputs.tensor[0])

        features = [self.poolers[p](features[p]) for p in self.backbone_features]

        features = torch.cat(features, dim=1)

        # the average poolers should cover the entirety of the receptive field
        assert [*features.shape[-2:]] == [1,1]

        features = features.view(-1, self.sum_feat_channels)

        features = self.output_network(features)

        out_dict = self.output_head.batch_to_terms_dicts(features)

        loss_dict = self.output_head.loss(out_dict, to_device(batched_inputs['outputs'],self.device))

        return loss_dict

    def preprocess_input(self, input_tensor):
        """
        Normalize, pad and batch the input images.
        """
        images = [ input_tensor[0].to(self.device) ]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images



