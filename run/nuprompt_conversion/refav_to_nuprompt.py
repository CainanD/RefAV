"""Script to convert the results from run_experiment.py to format used by NuPrompt's evaluation script."""
from pathlib import Path
import json
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
import traceback

from refAV.paths import NUSCENES_DIR, EXPERIMENTS, NUPROMPT_DATA_DIR
from run.nuprompt_conversion.nuscenes_to_av2 import NUSCENES_VAL_LOG_IDS

NUSCENES_TRACKING_CLASS_RANGES = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    }

def find_nuprompt_path(scene_token, combined_prompt):
    """
    Combined prompt is in the format "prompt1; prompt2; prompt3"
    """

    separated_prompts = combined_prompt.split(sep='; ')

    most_matching = 0
    best_path = None
    best_first_prompt = None
    for info_dict in scene_token_dict[scene_token]:
        gt_prompts = info_dict['prompts']

        num_matching = 0
        for prompt in separated_prompts:
            if prompt in gt_prompts:
                num_matching += 1
            
        if num_matching > most_matching:
            best_first_prompt = gt_prompts[0]
            best_path = info_dict['relative_path']
            most_matching = num_matching

    assert best_path is not None
    return best_path, best_first_prompt

def token(obj):
    """Convert list of dicts with 'token' key to dict indexed by token"""
    if isinstance(obj, dict) and 'sample_token' in obj:
        # Check if it's a list of dicts with 'token' keys
        return (obj['sample_token'], obj)
    if isinstance(obj, dict) and 'token' in obj:
        # Check if it's a list of dicts with 'token' keys
        return (obj['token'], obj)
    return obj

if __name__ == "__main__":

    exp_name ='exp70'
    with open(EXPERIMENTS, "rb") as file:
        exp_config = yaml.safe_load(file)
    tracker = exp_config[exp_name]["tracker"]
    split = exp_config[exp_name]["split"]

    nuscenes_tracker_path = Path('/home/crdavids/Trinity-Sync/PF-Track/ckpts/PF-Track-Models/f3_fullres_all/track_ext_5/results_nusc_tracking.json')
    #nuscenes_tracker_path = Path('/home/crdavids/Trinity-Sync/StreamPETR/tracking_results.json')

    pkl_path = Path(f'/home/crdavids/Trinity-Sync/RefAV/output/sm_predictions/{exp_name}/results/combined_predictions_{split}.pkl')
    tracking_predictions_dir = Path(f'/home/crdavids/Trinity-Sync/RefAV/output/tracker_predictions/{tracker}/{split}')

    with open(nuscenes_tracker_path, 'rb') as file:
        sp_results = json.load(file)['results']
    with open(pkl_path, 'rb') as file:
        refav_results = pickle.load(file)
    nuprompt_infos_val_path = Path('/home/crdavids/Trinity-Sync/PF-Track/nuprompt_infos_val.json')
    with open(nuprompt_infos_val_path, 'rb') as file:
        nuprompt_infos_val = json.load(file)['infos']
        # Extract unique prompt_filenames and normalize to match relative_nuprompt_path format
        nuprompt_val_paths = set()
        for info in nuprompt_infos_val:
            if 'prompt_filename' in info:
                # Convert ../data/nuscenes/... to ./data/nuscenes/...
                normalized_path = info['prompt_filename'].replace('../data/nuscenes/', './data/nuscenes/')
                nuprompt_val_paths.add(normalized_path)
        print(f'Loaded {len(nuprompt_val_paths)} unique prompt paths from nuprompt_infos_val.json')

    with open(NUSCENES_DIR/'sample.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        sample = {key: value for (key, value) in data_list}
        print('Loaded sample.json')
    with open(NUSCENES_DIR/'scene.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        scene = {key: value for (key, value) in data_list}
        print('Loaded scene.json')
    with open(NUSCENES_DIR/'sample_data.json', 'rb') as file:
        data_list = json.load(file)
        sample_data = {data['sample_token']:data for data in data_list if 'samples/LIDAR_TOP' in data['filename']}
        print('Loaded sample_data.json')
    with open(NUSCENES_DIR/'ego_pose.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        ego_pose = {key: value for (key, value) in data_list}
        print('Loaded ego_pose.json')
    with open(nuscenes_tracker_path, 'rb') as file:
        tracking_results = json.load(file)    

    sample_token_lookup = {}
    for sample_token, tracks in tqdm(tracking_results['results'].items()):

        scene_token = sample[sample_token]['scene_token']
        timestamp = int(sample[sample_token]['timestamp']/1e4)
        sample_token_lookup[(scene_token, timestamp)] = sample_token


    scene_token_dict:dict[str, list[dict[str,object]]] = {}
    for scene_dir in tqdm(list(NUPROMPT_DATA_DIR.iterdir())):
        scene_token = scene_dir.name
        if scene[scene_token]['name'] not in NUSCENES_VAL_LOG_IDS:
            continue

        if scene_token not in scene_token_dict:
            scene_token_dict[scene_token] = []

        for prompt_json in scene_dir.iterdir():
            with open(prompt_json, 'rb') as file:
                prompt_infos = json.load(file)

            infos_dict = {
                "relative_path": f'./data/nuscenes/nuprompt_v1.0/{scene_token}/{prompt_json.name}',
                "prompts": prompt_infos['prompt']
            }
            scene_token_dict[scene_token].append(infos_dict)    

    nuprompt_results = {}
    for (log_id, prompt), frames in tqdm(refav_results.items()):

        df = pd.read_feather(tracking_predictions_dir/log_id/'sm_annotations.feather')
        all_track_uuids = list(df['track_uuid'].unique())

        relative_nuprompt_path, first_prompt = find_nuprompt_path(log_id, prompt)
        tracking_name = f'{log_id}*{first_prompt}'
        if log_id == 'dce6f3f2bf6b4859abcf3268581969d3' and prompt == 'Walking; able to move; mobile':
            relative_nuprompt_path = './data/nuscenes/nuprompt_v1.0/dce6f3f2bf6b4859abcf3268581969d3/Walking.json'
            first_prompt = 'Walking'
            print(f'{prompt}: {tracking_name}')

        for frame in frames:
            timestamp_us = int(frame['timestamp_ns']/1e7)
            nuscenes_sample_token = sample_token_lookup[(log_id, timestamp_us)]
            nuprompt_sample_token = f'{nuscenes_sample_token}*{relative_nuprompt_path}'
            
            if nuprompt_sample_token not in nuprompt_results:
                nuprompt_results[nuprompt_sample_token] = []

            referred_box_uuids = []
            for i, uuid_index in enumerate(frame['track_id']):
                track_uuid = all_track_uuids[uuid_index]
                refav_category = frame['name'][i]
                if refav_category == 'REFERRED_OBJECT':
                    #try:
                    #    mask = (df['timestamp_ns'] == frame['timestamp_ns']) & (df['track_uuid'] == track_uuid)
                    #    distance = np.linalg.norm(df.loc[mask, ['tx_m', 'ty_m']].to_numpy())
                    #    category = df.loc[mask, 'category'].iloc[0]
                        #if distance > NUSCENES_TRACKING_CLASS_RANGES[category.lower()]:
                        #    print(f'Filtered distance: {distance}')
                        #    continue
                    #except: pass#traceback.print_exc()

                    referred_box_uuids.append(str(track_uuid))

            referred_boxes = []
            for sample_box in sp_results[nuscenes_sample_token]:
                if sample_box['tracking_id'] in referred_box_uuids:
                    #print(sample_box['tracking_id'])
                    referred_box = {
                        "sample_token":nuprompt_sample_token,
                        "translation":sample_box['translation'],
                        "size":sample_box['size'],
                        "rotation":sample_box['rotation'],
                        "velocity":sample_box['velocity'],
                        "tracking_name":tracking_name,
                        "attribute_name":"vehicle.moving",
                        "tracking_score":sample_box['tracking_score'],
                        "tracking_id":sample_box['tracking_id'],
                        "forecasting":np.zeros((4,2)).tolist(),
                    }
                    nuprompt_results[nuprompt_sample_token].append(referred_box)

    nuprompt_results_small = {}
    for nuprompt_sample_token, results in nuprompt_results.items():
        # Extract relative_nuprompt_path from the sample token (format: {sample_token}*{relative_path})
        relative_path = nuprompt_sample_token.split('*', 1)[1]
        if relative_path in nuprompt_val_paths:
            nuprompt_results_small[nuprompt_sample_token] = results
    
    print(f'NuPrompt results of length {len(nuprompt_results_small)}/300')

    nuprompt_format = {
        "meta":{
            "use_lidar":False,
            "use_camera":True,
            "use_radar":False,
            "use_map":False,
            "use_external":False
        },
        "results": nuprompt_results
    }
    nuprompt_format_small = {
        "meta":{
            "use_lidar":False,
            "use_camera":True,
            "use_radar":False,
            "use_map":False,
            "use_external":False
        },
        "results": nuprompt_results_small
    }

    # For use in the NuPrompt codebase evaluation script.
    with open(f'output/sm_predictions/{exp_name}/NuPrompt_results_{exp_name}_small.json', 'w') as file:
        json.dump(nuprompt_format_small, file, indent=4)

    with open(f'output/sm_predictions/{exp_name}/NuPrompt_results_{exp_name}.json', 'w') as file:
        json.dump(nuprompt_format, file, indent=4)