import argparse
import os
import subprocess
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

from dragonfly.apis.api_utils import preprocess_multifidelity_arguments
from dragonfly.exd.cp_domain_utils import load_config_file
from dragonfly.exd.experiment_caller import CPFunctionCaller
from dragonfly.opt import gp_bandit

sys.path.insert(0, './')
from easy_attributes.config import get_config as _get_config
from easy_attributes.utils.io import write_serialized, read_serialized


def mf_cost(z):
    return z[0] / 100.0


def setup_opt(config):
    domain = config.domain
    domain_orderings = config.domain_orderings
    (ask_tell_fidel_space, ask_tell_domain, _, ask_tell_mf_cost, ask_tell_fidel_to_opt, ask_tell_config, _) = \
        preprocess_multifidelity_arguments(config.fidel_space, domain, [mf_cost],
                                           mf_cost, config.fidel_to_opt, config)

    func_caller = CPFunctionCaller(None, ask_tell_domain, domain_orderings=domain_orderings,
                                   fidel_space=ask_tell_fidel_space, fidel_cost_func=ask_tell_mf_cost,
                                   fidel_to_opt=ask_tell_fidel_to_opt,
                                   fidel_space_orderings=config.fidel_space_orderings,
                                   config=ask_tell_config)

    opt = gp_bandit.CPGPBandit(func_caller, is_mf=True, ask_tell_mode=True)
    opt.initialise()
    return opt


PROC_METHODS = defaultdict(lambda: lambda x: str(x))
PROC_METHODS.update({"SOLVER.BASE_LR": lambda x: float(10.0 ** x),
                     "SOLVER.MOMENTUM": lambda x: float(1 - 10.0 ** x),
                     "SOLVER.ADAM_BETA": lambda x: float(1 - 10.0 ** x),
                     # "SOLVER.OPT_TYPE": lambda x: str(x),
                     "SOLVER.IMS_PER_BATCH": lambda x: int(2 ** x),
                     "SOLVER.MAX_TIME_SECS": lambda x: int(x),
                     "DATALOADER.NUM_WORKERS": lambda x: int(2 ** x)})


def process_draw(draw, draw_names):
    processed_draws = []
    draw_dict = {}
    for d, name in zip(draw, draw_names):
        proc_d = PROC_METHODS[name](d)
        processed_draws += [name, proc_d]
        draw_dict[name] = proc_d
    return processed_draws, draw_dict

def write_config(data_path: Path, draw_dict, draw_list,secs):
    cfg = _get_config(data_path,
                      use_mask=draw_dict['MASK_OR_BBOX'] == 'mask',
                      use_bounding_box=draw_dict['MASK_OR_BBOX'] != 'mask')

    cfg.SOLVER.CHECKPOINT_PERIOD = 100000000
    cfg.TEST.EVAL_PERIOD = 0
    cfg.SOLVER.MAX_ITER = int(18000 * secs *20 / (2400 * draw_dict["SOLVER.IMS_PER_BATCH"]))

    cfg.OUTPUT_DIR = 'bayesian_opt/output'
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    cfg.merge_from_list(draw_list)
    with open('bayesian_opt/output/bo_cfg.yml','w') as f:
        f.write(cfg.dump())
    # write_serialized(cfg,Path('bayesian_opt/output/cfg.yml'))

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--output_dir", type=str)
    # parser.add_argument('--model_weights', type=str)
    # parser.add_argument('--resume', action='store_true')
    # parser.add_argument("--distributed", action='store_true')
    # # parser.add_argument("--rank", type=int)
    # parser.add_argument('--debug', action='store_true')
    parser.add_argument('--input_data', type=str)
    # parser.add_argument('--use_mask_on_input', action='store_true')
    # parser.add_argument('--use_bounding_box_on_input', action='store_true')
    # parser.add_argument('--remove_cache', action='store_true')
    # parser.add_argument('--num_input_channels', type=int)
    return parser.parse_args()

def evaluate_objective():
    command = f'python -m easy_attributes.do_train --input_data /disk1/mcs_physics_data_derender/ --output_dir bayesian_opt/output --bo_config_file bayesian_opt/output/bo_cfg.yml --distributed'
    return os.system(command)

if __name__ == "__main__":
    config = load_config_file('bayesian_opt/params_domain.json')
    opt = setup_opt(config)
    args = parse_args()

    results = []
    while True:
        secs, draw = opt.ask()
        draw_list, draw_dict = process_draw(draw,
                                  config.domain_orderings.raw_name_ordering)
        write_config(Path(args.input_data),
                     draw_dict,
                     draw_list,
                     secs[0])
        # break
        # try:
        #     ret_code = evaluate_objective()
        # except Exception:
        #     ret_code = 1
        try:
            subprocess.run(['python', '-m', 'easy_attributes.do_train', '--input_data', '/disk1/mcs_physics_data_derender/', '--output_dir', 'bayesian_opt/output', '--bo_config_file', 'bayesian_opt/output/bo_cfg.yml', '--distributed'])
            ret_code = 0
        except:
            ret_code = 1
        # ret_code = 0
        if ret_code != 0 or not Path('bayesian_opt/output/last_results.json').exists():
            #failed
            results.append({'status': 'failed',
                            'draw': draw_dict,
                            'secs': int(secs[0]),
                            })
            opt.tell([(secs, draw, -1.0)])
        else:
            #didn't fail
            r = read_serialized(Path('bayesian_opt/output/last_results.json'))['results']
            results.append({'status':'successful',
                            'result':r,
                            'draw': draw_dict,
                            'secs': int(secs[0]),
                            })
            opt.tell([(secs, draw, -r['overall_mean'])])

        write_serialized(results, Path('bayesian_opt/bo_results.yml'))
        # break

