from pathlib import Path

import numpy as np
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from easy_attributes.utils.io import load_input


def visualize_data_dict(d, save_path: Path = None, channels=(0, 1, 2), input_tensor: torch.FloatTensor = None):
    # assumes d['file_name'] is in (C, H, W) format as required by detectron2
    # #the visualizer requires everything in rgb and (H, W, C) version
    img = input_tensor if input_tensor is not None else load_input(d['filename'])
    img = (img * 255)[list(channels)].numpy()
    if img.shape[0] == 1:
        img = np.concatenate([img] * 3, axis=0)
    assert img.shape[0] == 3
    img = img.swapaxes(0, 1).swapaxes(1, 2)
    img = img.astype(np.uint8)
    visualizer = Visualizer(img, metadata=MetadataCatalog.get('val'))
    out = visualizer.draw_dataset_dict({"annotations": [{"bbox": d['bbox'],
                                                         "bbox_mode": int(BoxMode.XYXY_ABS),
                                                         "segmentation": d['mask'],
                                                         "category_id": 0,
                                                         "iscrowd": 0}]}).get_image()
    if save_path:
        Image.fromarray(out).save(save_path)
    return out
