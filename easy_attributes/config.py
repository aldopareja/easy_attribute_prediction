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

    cfg.DATASETS.OUTPUTS = (
                            # 'agent_position_x',
                            # 'agent_position_y',
                            # 'agent_position_z',
                            # 'agent_rotation',
                            'dimension_0_x',
                            'dimension_0_y',
                            'dimension_0_z',
                            'dimension_1_x',
                            'dimension_1_y',
                            'dimension_1_z',
                            'dimension_2_x',
                            'dimension_2_y',
                            'dimension_2_z',
                            'dimension_3_x',
                            'dimension_3_y',
                            'dimension_3_z',
                            'dimension_4_x',
                            'dimension_4_y',
                            'dimension_4_z',
                            'dimension_5_x',
                            'dimension_5_y',
                            'dimension_5_z',
                            'dimension_6_x',
                            'dimension_6_y',
                            'dimension_6_z',
                            'dimension_7_x',
                            'dimension_7_y',
                            'dimension_7_z',
                            'position_x',
                            'position_y',
                            'position_z',
                            'rotation_x',
                            'rotation_y',
                            'rotation_z',
                            'shape',)

    cfg.DEBUG = debug

    metadata = read_serialized(data_path / 'metadata.yml')
    num_input_channels = metadata['inputs']['file_name']['num_channels']
    num_input_channels *= sum([use_mask, use_bounding_box]) + 1

    cfg.INPUT.FORMAT = "D" * num_input_channels
    cfg.MODEL.PIXEL_MEAN = [0.5] * num_input_channels
    cfg.MODEL.PIXEL_STD = [1.0] * num_input_channels

    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.FPN_OUT_FEATS = ('p2', 'p3', 'p4', 'p5', 'p6')
    cfg.MODEL.LAST_HIDDEN_LAYER_FEATS = 512

    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000  # a warm up is necessary to avoid diverging training while keeping the goal learning rate as high as possible
    cfg.SOLVER.IMS_PER_BATCH = 80 if not debug else 42
    # cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 80000
    # cfg.SOLVER.STEPS = (40000, 60000, 70000)
    # cfg.SOLVER.GAMMA = 0.5  # after each milestone in SOLVER.STEPS gets reached, the learning rate gets scaled by Gamma.
    cfg.SOLVER.BASE_LR = 6.658777172739463e-5

    cfg.SOLVER.OPT_TYPE = "Adam"  # options "Adam" "SGD"
    cfg.SOLVER.MOMENTUM = 0.9960477666835778  # found via Bayesian Optimization
    cfg.SOLVER.ADAM_BETA = 0.9999427846237621

    # cfg.SOLVER.WEIGHT_DECAY = 0.0005
    # cfg.SOLVER.WEIGHT_DECAY_BIAS = 0
    cfg.SOLVER.CHECKPOINT_PERIOD = 50 if debug else 2000  # 5000

    cfg.TEST.EVAL_PERIOD = 30 if debug else 4000

    return cfg

