from av2.evaluation.scenario_mining.eval import evaluate
from av2.datasets.sensor.splits import TEST, TRAIN, VAL
from av2.datasets.sensor.constants import AnnotationCategories

from utils import *
from refAV.scenario_prediction import predict_scenario_from_description
import pickle
import random
import json
from tqdm import tqdm
import time
import copy
import argparse
import glob
import re
import logging
import faulthandler


def evaluate_baseline(description,
                      log_id,
                      baseline_pred_dir:Path,
                      gt_pkl_dir, scenario_pred_dir):
    
    gt_pkl = gt_pkl_dir / log_id / f'{description}_{log_id[:8]}_ref_gt.pkl'
    pred_pkl = baseline_pred_dir / log_id / f'{description}_{log_id[:8]}_ref_predictions.pkl'

    if not pred_pkl.exists():
        pred_pkl = create_baseline_prediction(description, log_id, baseline_pred_dir, scenario_pred_dir)

    evaluate(pred_pkl, gt_pkl, objective_metric='HOTA', max_range_m=200, dataset_dir=None, out=str(log_dir/'eval'))


def create_baseline_prediction(description, log_id, baseline_pred_dir, scenario_pred_dir):
    
    #Used in exec(scenario) code
    is_gt = False
    output_dir = baseline_pred_dir 
    log_dir:Path = baseline_pred_dir / log_id
    
    scenario_filename = scenario_pred_dir / 'predicted_scenarios' / f'{description}.txt'
    print(scenario_filename)
    if scenario_filename.exists():
        print('Cached scenario prediction found')
    else:
        scenario_filename = predict_scenario_from_description(description, output_dir=scenario_pred_dir)
    
    with open(scenario_filename, 'r') as f:
        scenario = f.read()
        exec(scenario)

    pred_path = baseline_pred_dir / log_id / f'{description}_{log_id[:8]}_ref_predictions.pkl'
    return pred_path

def create_default_prediction(description, log_id, baseline_pred_dir):
    
    #Used in exec(scenario) code
    log_dir:Path = baseline_pred_dir / log_id
    
    empty_set = {}
    output_scenario(empty_set, description, log_dir, baseline_pred_dir, is_gt=False)

    pred_path = baseline_pred_dir / log_id / f'{description}_{log_id[:8]}_ref_predictions.pkl'
    if pred_path.exists():
        print('Default scenario prediction correctly generated.')
    else:
        print('Default scenario prediction failed.')

    return pred_path


def evaluate_pkls(pred_pkl, gt_pkl):
    evaluate(pred_pkl, gt_pkl, objective_metric='HOTA', max_range_m=100, dataset_dir=None, out='output/eval')


def clear_pkl_files(dir:Path):
    for file in dir.iterdir():
        if file.is_file() and file.suffix == '.pkl':
            file.unlink()
            print(f'{file.name} deleted')


def combine_matching_pkls(gt_base_dir, pred_base_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all log_ids from both directories
    gt_log_ids = {d.name: d for d in Path(gt_base_dir).iterdir() if d.is_dir()}
    pred_log_ids = {d.name: d for d in Path(pred_base_dir).iterdir() if d.is_dir()}
    
    # Find matching log_ids
    matching_log_ids = set(gt_log_ids.keys()) & set(pred_log_ids.keys())
    
    # Initialize combined dictionaries
    combined_gt = {}
    combined_pred = {}
    
    # For each matching log_id
    for log_id in matching_log_ids:

        gt_log_dir = gt_log_ids[log_id]
        pred_log_dir = pred_log_ids[log_id]
        
        # Get all PKL files in these directories
        gt_files = {f.stem.replace('_ref_gt', ''): f 
                   for f in gt_log_dir.glob('*_ref_gt.pkl')}
        pred_files = {f.stem.replace('_ref_predictions', ''): f 
                     for f in pred_log_dir.glob('*_ref_predictions.pkl')}
        
        # Find matching files within this log_id
        matching_keys = set(gt_files.keys()) & set(pred_files.keys())
        # Combine matching files
        for key in matching_keys:

            # Load GT file
            with open(gt_files[key], 'rb') as f:
                gt_data = pickle.load(f)
                combined_gt.update(copy.deepcopy(gt_data))
                
            # Load prediction file
            with open(pred_files[key], 'rb') as f:
                pred_data = pickle.load(f)
                combined_pred.update(copy.deepcopy(pred_data))

        # Report unmatched files for this log_id
        unmatched_gt = set(gt_files.keys()) - matching_keys
        unmatched_pred = set(pred_files.keys()) - matching_keys
        
        if unmatched_gt:
            print(f"\nUnmatched GT files in log_id {log_id}:")
            for name in unmatched_gt:
                print(f"- {name}")
        
        if unmatched_pred:
            print(f"\nUnmatched prediction files in log_id {log_id}:")
            for name in unmatched_pred:
                print(f"- {name}")
    
    # Save combined files
    if combined_gt:
        with open(os.path.join(output_dir, 'combined_gt.pkl'), 'wb') as f:
            pickle.dump(combined_gt, f)
    
    if combined_pred:
        with open(os.path.join(output_dir, 'combined_predictions.pkl'), 'wb') as f:
            pickle.dump(combined_pred, f)
    
    # Print statistics
    print(f"\nFound {len(matching_log_ids)} matching log_ids")
    print(f"Combined GT file contains {len(combined_gt)} entries")
    print(f"Combined predictions file contains {len(combined_pred)} entries")
    
    # Report unmatched log_ids
    unmatched_gt_logs = set(gt_log_ids.keys()) - matching_log_ids
    unmatched_pred_logs = set(pred_log_ids.keys()) - matching_log_ids
    
    if unmatched_gt_logs:
        print("\nLog IDs in GT without matching predictions directory:")
        for log_id in unmatched_gt_logs:
            print(f"- {log_id}")
    
    if unmatched_pred_logs:
        print("\nLog IDs in predictions without matching GT directory:")
        for log_id in unmatched_pred_logs:
            print(f"- {log_id}")
    
def file_contains_string(directory, search_string):
    return any(search_string in file.name for file in Path(directory).iterdir() if file.is_file())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--split", type=str, help="An optional argument", default='val')
    parser.add_argument("--start_log_index", type=int, help="An optional argument", default=0)
    parser.add_argument("--end_log_index", type=int, help="An optional argument", default=1000)
    parser.add_argument("--min_tp_descriptions_per_log", type=int, default=2)
    args = parser.parse_args()
    split = args.split

    faulthandler.enable()
    logging.basicConfig(
    filename='/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/generation_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if split == 'test':
        logs_to_analyze = list(TEST)[args.start_log_index:min(args.end_log_index, len(TEST))]
    elif split == 'train':
        logs_to_analyze = list(TRAIN)[args.start_log_index:min(args.end_log_index, len(TRAIN))]
    elif split == 'val':
        logs_to_analyze = list(VAL)[args.start_log_index:min(args.end_log_index, len(VAL))]
    else:
        print('--split must be one of train, test, or val.')

    output_dir = Path(f'/home/crdavids/Trinity-Sync/av2-api/output/pickles/{split}')
    gt_pkl_dir = output_dir
    baseline_pred_dir = Path(f'/home/crdavids/Trinity-Sync/av2-api/output/tracker_predictions/{split}')
    dataset_dir = Path(f'/data3/crdavids/refAV/dataset/{split}')

    existing_description_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/existing_descriptions.txt')
    scenario_def_dir = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/gt_scenarios')
    scenario_pred_dir = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation')
    eval_dir = Path('/home/crdavids/Trinity-Sync/av2-api/output/eval')

    cache_stats_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/cache_stats.json')
    log_stats_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/log_stats.json')
    description_stats_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/description_stats.json')
    
    logs_to_eval = sorted(list(baseline_pred_dir.iterdir()))[args.start_log_index:args.end_log_index]
    for log_dir in tqdm(logs_to_eval):
        if log_dir.is_dir() and log_dir.name:
            for scenario_description in Path(gt_pkl_dir/log_dir.name).iterdir():
                if "_ref_gt.pkl" in scenario_description.name:
                    try:
                        description = scenario_description.stem.split('_',1)[0]
                        if not file_contains_string(baseline_pred_dir/log_dir.name, description):
                            create_baseline_prediction(description, log_dir.name, baseline_pred_dir, scenario_pred_dir)
                        else:
                            print(f'Prediction pkl already found for {description}')
                    except:
                        logging.exception("An error occurred")
                        print(f'Evaluation of {scenario_description.stem} failed for log {log_dir.name}')
                        create_default_prediction(description, log_dir.name, baseline_pred_dir)

        evaluate_pkls(f'/home/crdavids/Trinity-Sync/av2-api/output/eval/{split}/combined_predictions.pkl',f'output/eval/{split}/combined_gt.pkl')

   #Do not use Trinity-1-4 or Trinity-1-34 to generate scenario visualizations. Nvidia drivers incompatible with Pyvista

    
