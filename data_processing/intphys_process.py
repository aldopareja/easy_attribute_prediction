import argparse
import os
import shutil
import sys
from itertools import chain
from pathlib import Path
from typing import Dict

import numpy as np
import hickle as hkl
from pycocotools import mask as mask_util
from PIL import Image

sys.path.insert(0, './')
from easy_attributes.utils.io import read_serialized, write_serialized
from easy_attributes.utils.istarmap_tqdm_patch import array_apply
from easy_attributes.utils.meta_data import get_discrete_metadata, get_pixels_mean_and_std

MIN_AREA = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    return args


def build_recursive_case_paths(input_folder, cases):
    if "scene" not in os.listdir(input_folder):
        to_recurse = sorted(list(os.path.join(input_folder, sub_folder) for sub_folder in os.listdir(input_folder)))
        for new_folder in to_recurse:
            if os.path.isdir(new_folder):
                build_recursive_case_paths(new_folder, cases)
    else:
        cases.append(Path(input_folder))
    return cases


def get_attributes(obj: Dict):
    attributes = {}

    attributes['shape'] = obj.get('shape', 'Occluder')

    # [attributes['position']]

    return attributes


def process_frame(depth_file, mask_file, rgb_file, frame, out_dir, index):
    depth_array = np.asarray(Image.open(depth_file), dtype=np.float32)
    # Intphys  encoding  here: https://www.intphys.com/benchmark/training_set.html
    depth_array = (2 ** 16 - 1 - depth_array) / 1000.0
    depth_array = 1 / (1 + depth_array)

    rgb = np.asarray(Image.open(rgb_file), dtype=np.float32) / 255.0

    input_array = np.concatenate([rgb, depth_array[..., np.newaxis]], axis=2)
    input_array = input_array.swapaxes(2, 1).swapaxes(1, 0)

    input_file = out_dir / 'inputs' / (str(index).zfill(9) + '.hkl')
    hkl.dump(input_array, input_file,
             mode='w', compression='gzip')

    masks = np.asarray(Image.open(mask_file))

    objects = []
    for oid, mask_val in frame['masks'].items():
        if not ('occluder' in oid or 'object' in oid):
            continue

        mask = masks == mask_val
        if mask.sum() < MIN_AREA:
            continue

        mask_y, mask_x = mask.nonzero()
        bbox = list(map(int, [mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()]))
        if bbox[3] <= bbox[1] + 2 and bbox[2] <= bbox[0] + 2:  # width and height shouldn't be too small
            continue

        mask = mask_util.encode(np.asarray(mask, order="F"))
        mask['counts'] = mask['counts'].decode('ascii')

        attributes = get_attributes(frame[oid])

        objects.append({'mask': mask,
                        'bbox': bbox,
                        **attributes,
                        'filename': str(input_file),
                        })

    return objects


def process_video(video_path, vid_num, out_dir):
    depths = []
    rgbs = []
    masks = []
    frames = []
    status = read_serialized(video_path / 'status.json')
    for i in range(1, 101):
        depths.append(video_path / 'depth' / ('depth_' + str(i).zfill(3) + '.png'))
        rgbs.append(video_path / 'scene' / ('scene_' + str(i).zfill(3) + '.png'))
        masks.append(video_path / 'masks' / ('masks_' + str(i).zfill(3) + '.png'))
        frames.append(status['frames'][i - 1])

    objects = [process_frame(d, m, r, f, out_dir, vid_num * 1000 + f_num) for f_num, (d, m, r, f) in
               enumerate(zip(depths, masks, rgbs, frames))]

    return chain.from_iterable(objects)


if __name__ == '__main__':
    args = parse_args()
    video_folders_val = build_recursive_case_paths(args.input_dir / 'dev_meta', [])
    video_folders_train = build_recursive_case_paths(args.input_dir / 'train', [])

    shutil.rmtree(args.output_dir, ignore_errors=True)
    (args.output_dir / 'inputs').mkdir(parents=True, exist_ok=True)

    data = {}
    for f_set, folders in {'val': video_folders_val, 'train': video_folders_train}.items():
        worker_args = [(v, i, args.output_dir) for i, v in enumerate(folders)]

        objects = list(chain.from_iterable(array_apply(process_video,
                                                       worker_args,
                                                       args.parallel,
                                                       # cpu_frac=2,
                                                       chunksize=10,
                                                       description='processing intphys scenes')))

        data[f_set] = objects

    meta_data = {'inputs': {'file_name': {'type': 'input_tensor',
                                          'num_channels': 4,
                                          'height': 288,
                                          'width': 288,
                                          **get_pixels_mean_and_std(data['val'])},
                            'mask': {'type': 'bitmask'},
                            'bbox': {'type': 'bounding_box'}},
                 'outputs': {'shape': get_discrete_metadata(data['val'] + data['train'],
                                                            'shape')}
                 }

    write_serialized(data['val'], args.output_dir / 'val.json')
    write_serialized(meta_data, args.output_dir / 'metadata.yml')
    write_serialized(data['train'], args.output_dir / 'train.json')
