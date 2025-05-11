import argparse
from pathlib import Path
import os
import yaml
import json
import refAV.paths as paths
from refAV.dataset_conversion import separate_scenario_mining_annotations, pickle_to_feather, create_gt_mining_pkls_parallel
from refAV.parallel_scenario_prediction import run_parallel_eval
from refAV.eval import evaluate_baseline

parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument("--exp_name", type=str, help="Enter the name of the experiment from experiments.yml you would like to run.")
args = parser.parse_args()

with open(paths.EXPERIMENTS, 'rb') as file:
    config = yaml.safe_load(file)

exp_name = config[args.exp]['name']
llm = config[args.exp]['LLM']
tracker= config[args.exp]['tracker']
split = config[args.exp]['split']

if llm not in config["LLM"]:
    print('Experiment uses an invalid LLM')
if tracker not in config["tracker"]:
    print('Experiment uses invalid tracking results')
if split not in ['train', 'test', 'val']:
    print('Experiment must use split train, test, or val')

if split in ['val', 'train']:
    sm_feather = Path(f'av2_sm_downloads/scenario_mining_{split}_annotations.feather')

    sm_data_split_path = paths.SM_DATA_DIR / split
    if not sm_data_split_path.exists():
        separate_scenario_mining_annotations(sm_feather, sm_data_split_path)
        create_gt_mining_pkls_parallel(sm_feather, sm_data_split_path, num_processes=max(1, int(.9*os.cpu_count())))

tracker_predictions_pkl = Path(f'tracker_downloads/{tracker}_{split}.pkl')
tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split

if not tracker_predictions_dest.exists():
    av2_data_split = paths.AV2_DATA_DIR / split
    pickle_to_feather(av2_data_split, tracker_predictions_pkl, tracker_predictions_dest)


log_index_start=0
if split == 'train':
     log_index_end=700
else:
    log_index_end=150
run_parallel_eval(exp_name, log_index_start, log_index_end)




