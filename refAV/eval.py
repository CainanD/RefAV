import pickle
import yaml
import json
import copy
import argparse
import logging
import faulthandler
import traceback
import os
from tqdm import tqdm
from pathlib import Path

from av2.evaluation.scenario_mining.eval import evaluate
from av2.datasets.sensor.splits import TEST, TRAIN, VAL
from refAV.utils import cache_manager, get_log_split
from refAV.code_generation import predict_scenario_from_description
from refAV.atomic_functions import *
import refAV.paths as paths


def evaluate_baseline(description,
                      log_id,
                      baseline_pred_dir:Path,
                      gt_pkl_dir, scenario_pred_dir):
    
    gt_pkl = gt_pkl_dir / log_id / f'{description}_{log_id[:8]}_ref_gt.pkl'
    pred_pkl = baseline_pred_dir / log_id / f'{description}_predictions.pkl'

    if not pred_pkl.exists():
        pred_pkl = create_baseline_prediction(description, log_id, baseline_pred_dir, scenario_pred_dir)

    evaluate(pred_pkl, gt_pkl, objective_metric='HOTA', max_range_m=50, dataset_dir=paths.AV2_DATA_DIR, out=str('eval'))


def execute_scenario(scenario, description, log_dir, output_dir:Path, is_gt=False):
    """Executes string as a python script in a local namespace."""
    exec(scenario)


def create_baseline_prediction(description:str, log_id:str, llm_name, tracker_name, experiment_name):

    split = get_log_split(log_id)
    pred_path = (paths.SM_PRED_DIR / experiment_name / split / log_id / f'{description}_predictions.pkl').resolve()
    if pred_path.exists():
        print(f'Cached scenario prediction exists.')
        return pred_path

    #Used in exec(scenario) code
    output_dir = paths.SM_PRED_DIR / experiment_name / split
    log_dir:Path = paths.TRACKER_PRED_DIR / tracker_name / split / log_id
    
    try:
        scenario_filename = paths.LLM_PRED_DIR / llm_name / f'{description}.txt'
        if scenario_filename.exists():
            print('Cached scenario definition found')
        else:
            scenario_filename = predict_scenario_from_description(description, output_dir=paths.LLM_PRED_DIR, model=llm_name)
        
        print(f'Evaluating log for {description}.')
        with open(scenario_filename, 'r') as f:
            scenario = f.read()
            execute_scenario(scenario, description, log_dir, output_dir)

    except Exception as e:
        # Sometimes the LLM will generate scenario definitions with bugs
        # In this case, output the default prediction of no referred tracks
        print(f"Error predicting scenario: {e}")
        traceback.print_exc()
        pred_path = create_default_prediction(description, log_dir, output_dir)

    return pred_path


def create_default_prediction(description, log_dir, output_dir):
    
    empty_set = {}
    output_scenario(empty_set, description, log_dir, output_dir, visualize=False)

    pred_path = output_dir / log_id / f'{description}_predictions.pkl'
    if pred_path.exists():
        print('Default scenario prediction correctly generated.')
    else:
        print('Default scenario prediction failed.')

    return pred_path


def evaluate_pkls(pred_pkl, gt_pkl):
    with open(pred_pkl, 'rb') as f:
        predictions = pickle.load(f)

    with open(gt_pkl, 'rb') as f:
        labels = pickle.load(f)

    for (log_id, prompt) in labels.keys():
        split = get_log_split(Path(log_id))
        break

    method_name = pred_pkl.name.split(sep='_')[0]

    output_dir = f'output/evaluation/{method_name}'
    metrics = evaluate(predictions, labels, objective_metric='HOTA', max_range_m=50, dataset_dir=paths.AV2_DATA_DIR/split, out=output_dir)

    metrics_dict = {
        'HOTA-Temporal': metrics[0],
        'HOTA': metrics[1],
        'Timestamp F1': metrics[2],
        'Log F1': metrics[3]
    }

    with open(f'{output_dir}/results.json', 'w') as f:
        json.dumps(f, metrics_dict, indent=4)

    return metrics_dict


def combine_matching_pkls(gt_base_dir, pred_base_dir, output_dir, method_name='ref'):
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
    for log_id in tqdm(matching_log_ids):

        gt_log_dir = gt_log_ids[log_id]
        pred_log_dir = pred_log_ids[log_id]
        
        # Get all PKL files in these directories
        gt_files = {f.stem.replace('_ref_gt', ''): f 
                   for f in gt_log_dir.glob('*_ref_gt.pkl')}
        pred_files = {f.stem.replace(f'_{method_name}_predictions', ''): f 
                     for f in pred_log_dir.glob(f'*_{method_name}_predictions.pkl')}
        
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
        with open(os.path.join(output_dir, f'{method_name}_predictions.pkl'), 'wb') as f:
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


def combine_pkls(gt_base_dir, pred_base_dir, output_dir, method_name='ref'):
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
    for log_id in tqdm(matching_log_ids):

        gt_log_dir = gt_log_ids[log_id]
        pred_log_dir = pred_log_ids[log_id]
        
        # Get all PKL files in these directories
        gt_files = {f.stem.replace('_ref_gt', ''): f 
                   for f in gt_log_dir.glob('*_ref_gt.pkl')}
        pred_files = {f.stem.replace(f'_{method_name}_predictions', ''): f 
                     for f in pred_log_dir.glob(f'*_{method_name}_predictions.pkl')}
        
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

        unmatched_gt = set(gt_files.keys()) - matching_keys

        for key in unmatched_gt:
            description = key.split(sep='_')[0]
            default_pred = create_default_prediction(description, log_id, pred_base_dir, method_name)

            with open(gt_files[key], 'rb') as f:
                gt_data = pickle.load(f)
                combined_gt.update(copy.deepcopy(gt_data))

            with open(default_pred, 'rb') as f:
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
        with open(os.path.join(output_dir, f'{method_name}_predictions.pkl'), 'wb') as f:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--start_log_index", type=int, help="An optional argument", default=0)
    parser.add_argument("--end_log_index", type=int, help="An optional argument", default=150)
    parser.add_argument("--num_processes", type=int, help="Number of parallel processes you want to use for computation", default=max(int(0.9 * os.cpu_count()), 1))
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    with open(paths.EXPERIMENTS, 'rb') as file:
        exp_config = yaml.safe_load(file)

    exp_name = exp_config[args.exp_name]['name']
    tracker_name= exp_config[args.exp_name]['tracker']
    llm_name = exp_config[args.exp_name]['LLM']
    split = exp_config[args.exp_name]['split']

    faulthandler.enable()
    logging.basicConfig(
    filename='output/evaluation_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cache_manager.num_processes = args.num_processes
    if split == 'test':
        logs_to_analyze = list(TEST)[args.start_log_index:min(args.end_log_index, len(TEST))]
    elif split == 'train':
        logs_to_analyze = list(TRAIN)[args.start_log_index:min(args.end_log_index, len(TRAIN))]
    elif split == 'val':
        logs_to_analyze = list(VAL)[args.start_log_index:min(args.end_log_index, len(VAL))]
    else:
        print('--split must be one of train, test, or val.')

    log_prompt_input_path = Path(f'av2_sm_downloads/log_prompt_pairs_{split}.json')
    eval_output_dir = Path(f'output/evaluation/{exp_name}/{split}')

    with open(log_prompt_input_path, 'rb') as f:
        log_prompts = json.load(f)

    for i, (log_id, prompts) in enumerate(log_prompts.items()):
        if log_id in logs_to_analyze:
            cache_manager.clear_all()
            for prompt in tqdm(prompts, desc=f'{i}/{len(logs_to_analyze)}'):
                create_baseline_prediction(prompt, log_id, llm_name, tracker_name, exp_name)
