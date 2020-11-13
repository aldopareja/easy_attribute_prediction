import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from detectron2.engine import default_setup, launch

from easy_attributes.attributes_dataset import build_datasets
from easy_attributes.config import get_config
from easy_attributes.trainer import CustomTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--model_weights', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    # parser.add_argument("--rank", type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--use_mask_on_input', action='store_true')
    parser.add_argument('--use_bounding_box_on_input', action='store_true')
    # parser.add_argument('--remove_cache', action='store_true')
    # parser.add_argument('--num_input_channels', type=int)
    return parser.parse_args()


def main(args):
    if args.debug and not args.distributed:
        import ipdb
        ipdb.set_trace()
    if not args.resume:
        shutil.rmtree(args.output_dir, ignore_errors=True)

    cfg = get_config(data_path=Path(args.input_data),
                     model_weights_path=Path(args.model_weights) if args.model_weights else None,
                     output_path=Path(args.output_dir) if args.output_dir else None,
                     debug=args.debug,
                     use_mask=args.use_mask_on_input,
                     use_bounding_box=args.use_bounding_box_on_input)

    default_setup(cfg, args)
    build_datasets(cfg, Path(args.input_data))
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(args.resume)
    trainer.train()

    # resume = args.model_weights is not None
    # trainer.resume_or_load(args.resume)
    # trainer.train()

if __name__ == "__main__":
    args = parse_args()
    num_gpus = torch.cuda.device_count()
    if args.distributed and num_gpus>1:
        port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
        launch(
            main,
            num_gpus,
            args= (args,),
            num_machines=1,
            dist_url= "tcp://127.0.0.1:{}".format(port),
            machine_rank=0
        )
    else:
        main(args)
