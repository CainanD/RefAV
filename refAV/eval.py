from av2.evaluation.scenario_mining.eval import evaluate
from av2.datasets.sensor.splits import TEST, TRAIN, VAL
from av2.datasets.sensor.constants import AnnotationCategories

from utils import *
from scenario_generation import predict_scenario_from_description, generate_scenarios
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


def get_max_run_number(directory):
    pattern = r'run(\d+)_description_stats\.json'
    max_run = -1
    
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            run_number = int(match.group(1))
            max_run = max(max_run, run_number)
    
    return max_run


def update_description_stats(folder_path='.', delete_run_files=True):
    """
    Updates the main description_stats.json file with data from all run files.
    The description_stats files have the format:
    {
        "description": {
            "average time to analyze": float,
            "positive_logs": [log_ids],
            "negative_logs": [log_ids],
            "failed_logs": [log_ids]
        }
    }
    
    Args:
        folder_path (str): Path to the folder containing JSON files
    """
    # Find all files
    main_file = os.path.join(folder_path, "description_stats.json")
    run_files = glob.glob(os.path.join(folder_path, "run*_description_stats.json"))
    
    print(f"Found main file: {main_file}")
    print(f"Found run files: {run_files}")
    
    # Load main file
    try:
        with open(main_file, 'r') as f:
            main_data = json.load(f)
    except FileNotFoundError:
        print(f"Main file {main_file} not found. Creating new file.")
        main_data = {}
    except json.JSONDecodeError:
        print(f"Error decoding {main_file}. Creating new data structure.")
        main_data = {}
    
    # Process all run files
    for run_file in run_files:
        try:
            with open(run_file, 'r') as f:
                run_data = json.load(f)
            
            # Update main data with run data
            for description, stats in run_data.items():
                if description not in main_data:
                    # If description doesn't exist in main, add it completely
                    main_data[description] = stats
                else:
                    # Merge list fields
                    for list_field in ["positive_logs", "negative_logs", "failed_logs"]:
                        if list_field in stats:
                            # Initialize field if not present
                            if list_field not in main_data[description]:
                                main_data[description][list_field] = []
                            
                            # Handle case where the field might be a float instead of a list
                            if isinstance(stats[list_field], list):
                                current_list = main_data[description][list_field]
                                if not isinstance(current_list, list):
                                    current_list = []
                                
                                # Add new log IDs without duplicates
                                main_data[description][list_field] = list(set(
                                    current_list + stats[list_field]
                                ))
                            else:
                                print(f"Warning: {list_field} in {description} is not a list in {run_file}")
                    
                    # Update average time if it exists in the run data
                    if "average time to analyze" in stats:
                        main_data[description]["average time to analyze"] = stats["average time to analyze"]
                        
            print(f"Processed {run_file}")
        except Exception as e:
            print(f"Error processing {run_file}: {e}")
    
    # Save updated main file
    with open(main_file, 'w') as f:
        json.dump(main_data, f, indent=4)
    
    print(f"Updated {main_file} successfully!")
    
    # Delete run files if requested
    if delete_run_files:
        for run_file in run_files:
            try:
                os.remove(run_file)
                print(f"Deleted {run_file}")
            except Exception as e:
                print(f"Error deleting {run_file}: {e}")


def update_log_stats(folder_path='.', delete_run_files=True):
    """
    Updates the main log_stats.json file with data from all run files.
    The log_stats files have the format:
    {
        "split": {
            "log_id": [descriptions]
        }
    }
    
    Args:
        folder_path (str): Path to the folder containing JSON files
    """
    # Find all files
    main_file = os.path.join(folder_path, "log_stats.json")
    run_files = glob.glob(os.path.join(folder_path, "run*_log_stats.json"))
    
    print(f"Found main file: {main_file}")
    print(f"Found run files: {run_files}")
    
    # Load main file
    try:
        with open(main_file, 'r') as f:
            main_data = json.load(f)
    except FileNotFoundError:
        print(f"Main file {main_file} not found. Creating new file.")
        main_data = {}
    except json.JSONDecodeError:
        print(f"Error decoding {main_file}. Creating new data structure.")
        main_data = {}
    
    # Process all run files
    for run_file in run_files:
        try:
            with open(run_file, 'r') as f:
                run_data = json.load(f)
            
            # Update main data with run data
            for split, logs in run_data.items():
                if split not in main_data:
                    main_data[split] = {}
                
                for log_id, descriptions in logs.items():
                    if log_id not in main_data[split]:
                        main_data[split][log_id] = descriptions
                    else:
                        # Merge descriptions without duplicates
                        main_data[split][log_id] = list(set(main_data[split][log_id] + descriptions))
                        
            print(f"Processed {run_file}")
        except Exception as e:
            print(f"Error processing {run_file}: {e}")
    
    # Save updated main file
    with open(main_file, 'w') as f:
        json.dump(main_data, f, indent=4)
    
    print(f"Updated {main_file} successfully!")
    
    # Delete run files if requested
    if delete_run_files:
        for run_file in run_files:
            try:
                os.remove(run_file)
                print(f"Deleted {run_file}")
            except Exception as e:
                print(f"Error deleting {run_file}: {e}")


def categories_are_valid_subset(file_path: Union[str, Path], log_categories: set[str]) -> bool:
    """
    Check if all annotation categories found in the file are present in the log categories.
    
    Args:
        file_path: Path to the text file to check
        log_categories: Set of categories found in the log
        
    Returns:
        bool: True if all categories found in file are present in log_categories
    """
    file_path = Path(file_path)
    all_possible_categories = set(category.value for category in AnnotationCategories)
    categories_in_file = set()
    
    try:
        with file_path.open('r') as f:
            content = f.read()
            # Find all annotation categories in the file
            for category in all_possible_categories:
                if category in content:
                    categories_in_file.add(category)
        
        # Check if categories found in file are a subset of log categories
        return categories_in_file.issubset(log_categories)
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False
    
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


    n = get_max_run_number(scenario_def_dir.parent) + 1
    update_description_stats(str(description_stats_path.parent))
    update_log_stats(str(log_stats_path.parent))
    
    with open(log_stats_path, 'r') as file:
        log_stats = json.load(file)
    with open(description_stats_path, 'r') as file:
        description_stats = json.load(file)
    
    with open(scenario_def_dir.parent/f'run{n}_log_stats.json', "w") as file:
        json.dump(log_stats, file, indent=4)  # `indent=4` makes it human-readable
    with open(scenario_def_dir.parent/f'run{n}_description_stats.json', "w") as file:
        json.dump(description_stats, file, indent=4)  # `indent=4` makes it human-readable

    with open('/home/crdavids/Trinity-Sync/av2-api/output/scenario_generation/categories_in_log.json', 'r') as file:
        categories_per_log = json.load(file)


    logs_complete = False
    while not logs_complete:

        is_gt = True
        random.shuffle(logs_to_analyze)
        for log_idx, log_id in enumerate(logs_to_analyze):
            log_dir = dataset_dir / log_id

            if log_id not in logs_to_analyze:
                continue

            if log_id not in log_stats[split]:
                log_stats[split][log_id] = []

            categories_in_log = categories_per_log[log_dir.name]
            scenario_definitions = list(scenario_def_dir.iterdir())
            random.shuffle(scenario_definitions)
            for scenario_def in tqdm(scenario_definitions, desc=f'log_id {log_idx}/{len(logs_to_analyze)}'):
                if not scenario_def.is_file() or not categories_are_valid_subset(scenario_def, categories_in_log):
                    continue

                if len(log_stats[split][log_dir.name]) >= args.min_tp_descriptions_per_log:
                    break

                scenario_description = scenario_def.stem
                if scenario_description not in description_stats:
                    description_stats[scenario_description] = {}
                    description_stats[scenario_description]['average time to analyze'] = 0
                    description_stats[scenario_description]['positive_logs'] =  []
                    description_stats[scenario_description]['negative_logs'] = []
                    description_stats[scenario_description]['failed_logs'] =   []
            
                if ((len(description_stats[scenario_description]['positive_logs']) >= 700
                or len(description_stats[scenario_description]['failed_logs']) > 5)
                or log_id in description_stats[scenario_description]['positive_logs']
                or log_id in description_stats[scenario_description]['negative_logs']):
                    continue

                try:
                    start_time = time.time() 
                    with open(scenario_def, 'r') as f:
                        scenario = f.read()
                        exec(scenario)

                    end_time = time.time()

                    num_logs_analyzed = (len(description_stats[scenario_description]['positive_logs']) 
                                        + len(description_stats[scenario_description]['negative_logs'])
                                        + len(description_stats[scenario_description]['positive_logs']))
                    total_time = description_stats[scenario_description]['average time to analyze']*num_logs_analyzed+(end_time-start_time)
                    description_stats[scenario_description]['average time to analyze'] = total_time/(num_logs_analyzed+1)
                    
                    scenario_vis_dir = output_dir/log_dir.name/'scenario visualizations'
                    if scenario_vis_dir.exists() and file_contains_string(scenario_vis_dir, scenario_description):
                        log_stats[split][log_id].append(scenario_description)
                        description_stats[scenario_description]['positive_logs'].append(log_id)
                        
                        if scenario_description in description_stats[scenario_description]['failed_logs']:
                            description_stats[scenario_description]['failed_logs'].remove(log_id)
                    else:
                        empty_pkl = output_dir/log_dir.name/f'{scenario_description}_{log_dir.name[:8]}_ref_gt.pkl'
                        empty_pkl.unlink(missing_ok=True)
                        description_stats[scenario_description]['negative_logs'].append(log_id)

                except Exception as e:
                    print(f"An error at log id {log_dir.name}")
                    logging.exception(f"An error at log id {log_id} for description {scenario_description}")
                    description_stats[scenario_description]['failed_logs'].append(log_id)

                with open(scenario_def_dir.parent/f'run{n}_log_stats.json', "w") as file:
                    json.dump(log_stats, file, indent=4)  # `indent=4` makes it human-readable
                with open(scenario_def_dir.parent/f'run{n}_description_stats.json', "w") as file:
                    json.dump(description_stats, file, indent=4)  # `indent=4` makes it human-readable
                with open(cache_stats_path, 'w') as file:
                    json.dump(cache_manager.get_stats(), file, indent=4)

                if (len(description_stats[scenario_description]['failed_logs']) > 5
                and len(description_stats[scenario_description]['positive_logs']) + len(description_stats[scenario_description]['negative_logs']) == 0):
                    scenario_def.unlink()

        logs_complete = True
        for log_id, postive_descriptions in log_stats[split].items():
            if log_id in logs_to_analyze and len(postive_descriptions) < args.min_tp_descriptions_per_log:
                logs_complete = False
                break
    
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

    
    #combine_matching_pkls(output_dir, baseline_pred_dir, output_dir=f'output/eval/{split}')
    evaluate_pkls(f'/home/crdavids/Trinity-Sync/av2-api/output/eval/{split}/combined_predictions.pkl',f'output/eval/{split}/combined_gt.pkl')

   #Do not use Trinity-1-4 or Trinity-1-34 to generate scenario visualizations. Nvidia drivers incompatible with Pyvista

    
