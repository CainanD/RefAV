"""Script to reproduce RefProg results. It may take several hours to complete an experiment, depending on the number of tracks coming from the tracker."""
import argparse
import json
from pathlib import Path
import os
import yaml

import refAV.paths as paths
from refAV.dataset_conversion import (
    separate_scenario_mining_annotations,
    pickle_to_feather,
    create_gt_mining_pkls_parallel,
)
from refAV.parallel_scenario_prediction import run_parallel_eval
from refAV.eval import evaluate_pkls, combine_pkls
from refAV.utils import construct_caches

parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument(
    "--exp_name",
    type=str,
    default="exp4",
    help="Enter the name of the experiment from experiments.yml you would like to run.",
)
parser.add_argument(
    "--procs_per_task",
    type=int,
    default=3,
    help="The number of processes your eval script should launch with",
)
args = parser.parse_args()

with open(paths.EXPERIMENTS, "rb") as file:
    config = yaml.safe_load(file)

exp_name = config[args.exp_name]["name"]
llm = config[args.exp_name]["LLM"]
tracker = config[args.exp_name]["tracker"]
split = config[args.exp_name]["split"]

if llm not in config["LLM"]:
    print("Experiment uses an invalid LLM")
if tracker not in config["tracker"]:
    print("Experiment uses invalid tracking results")
if split not in ["train", "test", "val"]:
    print("Experiment must use split train, test, or val")

if split in ["train", "val"]:
    sm_feather = paths.SM_DOWNLOAD_DIR / f"scenario_mining_{split}_annotations.feather"

    sm_data_split_path = paths.SM_DATA_DIR / split
    if not sm_data_split_path.exists():
        separate_scenario_mining_annotations(sm_feather, sm_data_split_path)
        create_gt_mining_pkls_parallel(
            sm_feather,
            sm_data_split_path,
            num_processes=max(1, int(0.9 * os.cpu_count())),
        )

tracker_predictions_pkl = Path(f"tracker_downloads/{tracker}_{split}.pkl")
tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split

if not tracker_predictions_dest.exists():
    av2_data_split = paths.AV2_DATA_DIR
    pickle_to_feather(av2_data_split, tracker_predictions_pkl, tracker_predictions_dest)

log_prompts_path = paths.SM_DOWNLOAD_DIR / f"log_prompt_pairs_{split}.json"

# Build caches before parallel eval so subprocesses just load from disk
with open(log_prompts_path, 'rb') as f:
    log_prompts = json.load(f)
all_log_dirs = [paths.TRACKER_PRED_DIR / tracker / split / log_id for log_id in log_prompts.keys()]
construct_caches(all_log_dirs)

run_parallel_eval(exp_name, log_prompts_path, args.procs_per_task)

experiment_dir = paths.SM_PRED_DIR / exp_name 
combined_preds = combine_pkls(sm_data_split_path, log_prompts_path)#combine_pkls(experiment_dir / "scenario_predictions", log_prompts_path, suffix="_predictions")

# Only train and val splits will be evaluated
# if split in ["train", "val"]:
combined_gt = Path('/home/crdavids/Trinity-Sync/RefAV/scenario_mining_downloads/combined_gt_val.pkl')#combine_pkls(sm_data_split_path, log_prompts_path)
metrics = evaluate_pkls(combined_preds, combined_gt, experiment_dir)
print(metrics)
#else:
# print(Only train and val splits can be evaluated)
