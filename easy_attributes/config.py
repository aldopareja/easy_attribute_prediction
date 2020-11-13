from pathlib import Path

from detectron2.config import get_cfg as detectron_get_cfg
from detectron2.model_zoo import model_zoo

from easy_attributes.utils.io import read_serialized


def get_config(data_path: Path,
               model_weights_path: Path = None,
               output_path: Path = None,
               debug: bool = True,
               use_mask=True,
               use_bounding_box=True):
    cfg = detectron_get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))

    cfg.MODEL.META_ARCHITECTURE = 'CustomModel'

    if model_weights_path is None:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    else:
        cfg.MODEL.WEIGHTS = str(model_weights_path)

    cfg.OUTPUT_DIR = str(output_path) if output_path is not None else './output'
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)

    cfg.DATALOADER.NUM_WORKERS = 0 if debug else 6

    cfg.DATASETS.TRAIN = ("val",) if debug else ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATASETS.INPUTS = ('file_name',)
    cfg.DATASETS.INPUTS += ('mask',) if use_mask else ()
    cfg.DATASETS.INPUTS += ('bbox',) if use_bounding_box else ()

    cfg.DATASETS.OUTPUTS = ('shape', 'position_x')

    cfg.DEBUG = debug

    metadata = read_serialized(data_path / 'metadata.yml')
    num_input_channels = metadata['inputs']['file_name']['num_channels']
    num_input_channels *= sum([use_mask, use_bounding_box]) + 1

    cfg.INPUT.FORMAT = "D" * num_input_channels
    cfg.MODEL.PIXEL_MEAN = [0.5] * num_input_channels
    cfg.MODEL.PIXEL_STD = [1.0] * num_input_channels

    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.FPN_OUT_FEATS = ('p3', 'p4', 'p5')
    cfg.MODEL.LAST_HIDDEN_LAYER_FEATS = 512

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000  # a warm up is necessary to avoid diverging training while keeping the goal learning rate as high as possible
    cfg.SOLVER.IMS_PER_BATCH = 16 if not debug else 8
    # cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 80000
    # cfg.SOLVER.STEPS = (40000, 60000, 70000)
    # cfg.SOLVER.GAMMA = 0.5  # after each milestone in SOLVER.STEPS gets reached, the learning rate gets scaled by Gamma.

    return cfg
