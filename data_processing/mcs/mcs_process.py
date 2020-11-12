"""
This tool processes passive data for eval 3 and dumps it in a valid format to train a derender
"""
import argparse
import os
import random
import shutil
import sys
from itertools import repeat, chain, product
from multiprocessing import Process, Queue
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import hickle as hkl
from pycocotools import mask as mask_util
from PIL import Image

import machine_common_sense as mcs

sys.path.insert(0, './')
from easy_attributes.utils.io import write_serialized
from easy_attributes.utils.meta_data import get_continuous_metadata, get_discrete_metadata

MIN_AREA = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mcs_executable', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_path', type=str)
    # parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_parallel_controllers', type=int)
    args = parser.parse_args()
    args.data_path = Path(args.data_path)
    args.input_dir = Path(args.mcs_executable)
    args.output_dir = Path(args.output_dir)
    return args


def get_attributes(obj: mcs.object_metadata.ObjectMetadata):
    attributes = {}

    attributes['shape'] = obj.shape

    [attributes.__setitem__('position_' + k, v) for k, v in obj.position.items()]

    [attributes.__setitem__('rotation_' + k, v) for k, v in obj.rotation.items()]

    [attributes.__setitem__(f'dimension_{i}_{c}', obj.dimensions[i][c]) for i,c in product(range(8), 'xyz')]

    return attributes


def dump_for_detectron(step_data, out_path, index):
    # print(step_data)
    depth: np.ndarray = step_data.depth_map_list[0]
    depth = 1 / (1 + depth)
    rgb = np.array(step_data.image_list[0], dtype=np.float32) / 255.0
    input = np.concatenate([rgb, depth[..., np.newaxis]], axis=2)
    input = input.swapaxes(2, 0).swapaxes(1, 2)  # now it is C, H, W

    input_to_file = out_path / 'inputs' / (str(index).zfill(9) + '.hkl')
    hkl.dump(input, input_to_file, mode='w', compression='gzip')

    masks = np.array(step_data.object_mask_list[0])
    masks = masks[:, :, 0] + masks[:, :, 1] * 256 + masks[:, :, 2] * 256 ** 2
    assert not (masks == 0).any()

    foreground_objects = {e.color['r'] + e.color['g'] * 256 + e.color['b'] * 256 ** 2: e
                          for e in step_data.structural_object_list
                          if not (e.uuid.startswith('wall') or e.uuid.startswith('floor'))}

    foreground_objects.update({e.color['r'] + e.color['g'] * 256 + e.color['b'] * 256 ** 2: e
                               for e in step_data.object_list})

    objects = []
    for v in foreground_objects.keys():
        mask = masks == v
        if mask.sum() < MIN_AREA:
            continue

        mask_y, mask_x = mask.nonzero()
        bbox = list(map(int, [mask_x.min(), mask_y.min(), mask_x.max(), mask_y.max()]))
        if bbox[3] <= bbox[1] + 2 and bbox[2] <= bbox[0] + 2:  # width and height shouldn't be too small
            continue

        mask = mask_util.encode(np.asarray(mask, order="F"))
        mask['counts'] = mask['counts'].decode('ascii')
        attributes = get_attributes(foreground_objects[v])

        objects.append({'mask': mask,
                        'bbox': bbox,
                        **attributes,
                        'filename': str(input_to_file),
                        **{'agent_position_' + k: v for k, v in step_data.position.items()},
                        'agent_rotation': step_data.rotation})

    return objects


def process_scene(controller, scene_path, output_path, vid_index, concurrent, tp: ThreadPoolExecutor):
    config_data, _ = mcs.load_config_json_file(scene_path)

    jobs = []
    frame_id = 0
    step_data = controller.start_scene(config_data)
    if concurrent:
        jobs.append(tp.submit(dump_for_detectron, step_data,
                              output_path,
                              vid_index * 500 + frame_id))
    else:
        jobs.append(dump_for_detectron(step_data, output_path, vid_index * 500 + frame_id))

    frame_id += 1
    actions = config_data['goal']['action_list']
    for a in actions:
        assert len(a) == 1, "there must be an action"
        step_data = controller.step(a[0])
        if concurrent:
            jobs.append(tp.submit(dump_for_detectron, step_data,
                                  output_path,
                                  vid_index * 500 + frame_id))
        else:
            jobs.append(dump_for_detectron(step_data, output_path, vid_index * 500 + frame_id))

        frame_id += 1

    controller.end_scene("classification", 0.0)
    if concurrent:
        jobs = [j.result() for j in jobs]
    return chain.from_iterable(jobs)


class SequentialSceneProcessor:
    def __init__(self, mcs_executable: Path, concurrent_dump: bool):
        self.controller = mcs.create_controller(str(mcs_executable),
                                                depth_maps=True,
                                                object_masks=True,
                                                history_enabled=False)

        self.concurrent = concurrent_dump
        self.tp = ThreadPoolExecutor(4)

    def process(self, w_arg):
        (s, _, o, v) = w_arg
        return process_scene(self.controller, s, o, v, self.concurrent, self.tp)


def ParallelSceneProcess(work_q: Queue, result_q: Queue, mcs_executable: Path, concurrent_dump):
    controller = mcs.create_controller(str(mcs_executable),
                                       depth_maps=True,
                                       object_masks=True,
                                       history_enabled=False)
    with ThreadPoolExecutor(4) as p:
        while True:
            w_arg = work_q.get()
            if w_arg is None:
                break
            (s, _, o, v) = w_arg
            results = process_scene(controller, s, o, v, concurrent_dump, p)
            result_q.put(results)


if __name__ == "__main__":
    args = parse_args()
    scene_files = [args.data_path / a for a in args.data_path.iterdir()]

    shutil.rmtree(args.output_dir, ignore_errors=True)

    (args.output_dir / 'inputs').mkdir(parents=True, exist_ok=True)

    w_args = [(s, e, o, i) for i, (s, e, o) in enumerate(zip(scene_files,
                                                             repeat(args.mcs_executable),
                                                             repeat(args.output_dir)))]


    if args.num_parallel_controllers > 0:
        work_queue = Queue()
        result_queue = Queue()

        workers = [Process(target=ParallelSceneProcess,
                           args=(work_queue, result_queue, args.mcs_executable, True)) for _ in
                   range(args.num_parallel_controllers)]
        [w.start() for w in workers]

        w_args = [work_queue.put(w) for w in w_args]

        data_dicts = [result_queue.get() for _ in range(len(w_args))]

        [work_queue.put(None) for _ in range(args.num_parallel_controllers)]
        [w.join() for w in workers]
        work_queue.close()
        result_queue.close()
    else:
        worker = SequentialSceneProcessor(args.mcs_executable, False)
        data_dicts = [worker.process(w_arg) for w_arg in w_args]

    data_dicts = list(chain.from_iterable(data_dicts))

    all_indices = set(range(len(data_dicts)))
    val_indices = random.sample(all_indices, 6000)
    train_indices = all_indices.difference(val_indices)

    val_dicts = [data_dicts[i] for i in val_indices]
    train_dicts = [data_dicts[i] for i in train_indices]

    meta_data = {'inputs': {'file_name': {'type': 'input_tensor',
                                          'num_channels': 4,
                                          'height': 400,
                                          'width': 600},
                            'mask': {'type': 'bitmask'},
                            'bbox': {'type': 'bounding_box'}},
                 'outputs': {**{e: get_continuous_metadata(val_dicts, e)
                                for e in [*[c[0] + c[1] for c in product(['rotation_', 'position_', 'agent_position_'],
                                                                         'xyz')],
                                          *[f'dimension_{c[0]}_{c[1]}' for c in product(range(8), 'xyz')],
                                          'agent_rotation']},
                             'shape': get_discrete_metadata(data_dicts, 'shape')}
                 }

    write_serialized(val_dicts, args.output_dir / 'val.json')
    write_serialized(train_dicts, args.output_dir / 'train.json')
    write_serialized(meta_data, args.output_dir / 'metadata.yml')

    # kill stalling controllers
    os.system('pkill -f MCS-AI2-THOR-Unity-App -9')
