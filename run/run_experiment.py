import argparse
from pathlib import Path
import os
import yaml
import refAV.paths as paths
from refAV.dataset_conversion import separate_scenario_mining_annotations, pickle_to_feather, create_gt_mining_pkls_parallel
from refAV.parallel_scenario_prediction import run_parallel_eval
from refAV.eval import evaluate_pkls, combine_pkls

parser = argparse.ArgumentParser(description="Example script with arguments")
parser.add_argument("--exp_name", type=str, default='exp53', help="Enter the name of the experiment from experiments.yml you would like to run.")
parser.add_argument("--procs_per_task", type=int, default=3, help="The number of processes your eval script should launch with")
parser.add_argument("--dataset-root", type=Path, help="Directory containing version-specific annotations and log_prompt_pairs files")
parser.add_argument("--sensor-root", type=Path, help="Argoverse sensor dataset directory")
parser.add_argument("--ground-truth", type=Path, help="Trusted organizer ground-truth pickle for local evaluation")
parser.add_argument("--submission-format", choices=["json", "json.gz"], default="json.gz")
parser.add_argument("--temporal-only", action="store_true", help="Export timestamp booleans instead of tracks")
parser.add_argument("--predictions-only", action="store_true", help="Skip local evaluation (e.g. hidden test labels)")
args = parser.parse_args()
if not args.predictions_only and args.ground_truth is None:
    parser.error("--ground-truth is required for local evaluation; use --predictions-only for export")
if args.dataset_root:
    paths.SM_DOWNLOAD_DIR = args.dataset_root
if args.sensor_root:
    paths.AV2_DATA_DIR = args.sensor_root

with open(paths.EXPERIMENTS, 'rb') as file:
    config = yaml.safe_load(file)

exp_name = config[args.exp_name]['name']
llm = config[args.exp_name]['LLM']
tracker= config[args.exp_name]['tracker']
split = config[args.exp_name]['split']

if llm not in config["LLM"]:
    print('Experiment uses an invalid LLM')
if tracker not in config["tracker"]:
    print('Experiment uses invalid tracking results')
if split not in ['train', 'test', 'val']:
    print('Experiment must use split train, test, or val')

if split in ['val', 'train'] and not args.predictions_only:
    sm_feather = paths.SM_DOWNLOAD_DIR / f'scenario_mining_{split}_annotations.feather'

    sm_data_split_path = paths.SM_DATA_DIR / split
    if not sm_data_split_path.exists():
        separate_scenario_mining_annotations(sm_feather, sm_data_split_path)
        create_gt_mining_pkls_parallel(sm_feather, sm_data_split_path, num_processes=max(1, int(.9*os.cpu_count())))

tracker_predictions_pkl = Path(f'tracker_downloads/{tracker}_{split}.pkl')
tracker_predictions_dest = paths.TRACKER_PRED_DIR / tracker / split

if not tracker_predictions_dest.exists():
    av2_data_split = paths.AV2_DATA_DIR / split
    pickle_to_feather(av2_data_split, tracker_predictions_pkl, tracker_predictions_dest)

log_prompts_path = paths.SM_DOWNLOAD_DIR / f'log_prompt_pairs_{split}.json'
run_parallel_eval(exp_name, log_prompts_path, args.procs_per_task)

experiment_dir = paths.SM_PRED_DIR / exp_name
combined_preds = combine_pkls(experiment_dir, log_prompts_path)

from refAV.submission_format.cli import write_predictions
import pickle
with combined_preds.open("rb") as stream:
    exported = pickle.load(stream)  # Internal, organizer-created cache only.
if args.temporal_only:
    exported = {key: [{"timestamp_ns": int(frame["timestamp_ns"]),
                      "is_positive": bool(frame.get("is_positive", False) or
                                          (frame["label"] == 0).any())}
                     for frame in frames] for key, frames in exported.items()}
submission = experiment_dir / "results" / f"predictions_{split}.{args.submission_format}"
write_predictions(exported, submission)
print(f"Submission written to {submission}")
if not args.predictions_only:
    if args.ground_truth is None:
        parser.error("--ground-truth is required for local evaluation; use --predictions-only for export")
    metrics = evaluate_pkls(combined_preds, args.ground_truth, experiment_dir)
    print(metrics)
