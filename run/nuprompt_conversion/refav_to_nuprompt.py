"""Script to convert the results from run_experiment.py to format used by NuPrompt's evaluation script."""

from pathlib import Path
import json
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

NUPROMPT_DATA_PATH = Path('/home/crdavids/Trinity-Sync/Prompt4Driving/data/nuscenes/nuprompt_v1.0')
NUSCENES_PATH = Path('/home/crdavids/Trinity-Sync/Prompt4Driving/data/nuscenes/v1.0-trainval')

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

val = \
    ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
     'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
     'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
     'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
     'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
     'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
     'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
     'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
     'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
     'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
     'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
     'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
     'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
     'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
     'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
     'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
     'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
     'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
     'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']

class_range = {
    "car": 50,
    "truck": 50,
    "bus": 50,
    "trailer": 50,
    "pedestrian": 40,
    "motorcycle": 40,
    "bicycle": 40,
    "ego_vehicle": 50
    }

with open(NUSCENES_PATH/'sample.json', 'rb') as file:
    data_list = json.load(file, object_hook=token)
    sample = {key: value for (key, value) in data_list}
    print('Loaded sample.json')
with open(NUSCENES_PATH/'scene.json', 'rb') as file:
    data_list = json.load(file, object_hook=token)
    scene = {key: value for (key, value) in data_list}
    print('Loaded scene.json')
with open(NUSCENES_PATH/'sample_data.json', 'rb') as file:
    data_list = json.load(file)
    sample_data = {data['sample_token']:data for data in data_list if 'samples/LIDAR_TOP' in data['filename']}
    print('Loaded sample_data.json')
with open(NUSCENES_PATH/'ego_pose.json', 'rb') as file:
    data_list = json.load(file, object_hook=token)
    ego_pose = {key: value for (key, value) in data_list}
    print('Loaded ego_pose.json')
with open('/home/crdavids/Trinity-Sync/StreamPETR/tracking_results.json', 'rb') as file:
    tracking_results = json.load(file)    

sample_token_lookup = {}
for sample_token, tracks in tqdm(tracking_results['results'].items()):

    scene_token = sample[sample_token]['scene_token']
    timestamp = int(sample[sample_token]['timestamp']/1e4)
    sample_token_lookup[(scene_token, timestamp)] = sample_token


scene_token_dict:dict[str, list[dict[str,object]]] = {}
for scene_dir in tqdm(list(NUPROMPT_DATA_PATH.iterdir())):
    scene_token = scene_dir.name
    if scene[scene_token]['name'] not in val:
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


# Convert AV2 pickle file with StreamPETR referential tracking results to NuPrompt json file format.
# 1. load the pickle file and StreamPETR tracking JSON
#    for each log-prompt pair:
#       find the associated NuPrompt path
#       write to the NuPrompt json with list[
#           "log*propmt":{ StreamPETR bbox for each bbox with track_id that matches REFERRED_OBJECT track_id},...]
#                 

exp_name ='exp64'
stream_petr_path = Path('/home/crdavids/Trinity-Sync/StreamPETR/tracking_results.json')
prompt_track_path = Path('/home/crdavids/Trinity-Sync/Prompt4Driving/work_dirs/f3_prompttrack_nuprompt/results_prompt_tracking.json')
pkl_path = Path(f'/home/crdavids/Trinity-Sync/refbot/output/sm_predictions/{exp_name}/results/combined_predictions_nuprompt_val_large.pkl')
tracking_predictions_dir = Path('/home/crdavids/Trinity-Sync/refbot/output/tracker_predictions/StreamPETR_Tracking/nuprompt_val_large')

with open(stream_petr_path, 'rb') as file:
    sp_results = json.load(file)['results']
with open(pkl_path, 'rb') as file:
    refav_results = pickle.load(file)
with open(prompt_track_path, 'rb') as file:
    pt_results = json.load(file)['results']


nuprompt_results = {}
for (log_id, prompt), frames in tqdm(refav_results.items()):

    df = pd.read_feather(tracking_predictions_dir/log_id/'sm_annotations.feather')

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
        
        if nuscenes_sample_token not in nuprompt_results:
            nuprompt_results[nuprompt_sample_token] = []

        referred_box_ids = []
        for i, tracking_id in enumerate(frame['track_id']):
            refav_category = frame['name'][i]
            if refav_category == 'REFERRED_OBJECT':
                try:
                    mask = (df['timestamp_ns'] == frame['timestamp_ns']) & (df['track_uuid'] == tracking_id)
                    distance = np.linalg.norm(df.loc[mask, ['tx_m', 'ty_m']].to_numpy())
                    category = df.loc[mask, 'category'].iloc[0]
                    if distance > class_range[category.lower()]:
                        print(f'Filtered distance: {distance}')
                        continue
                except: pass

                referred_box_ids.append(tracking_id)

        referred_boxes = []
        for sample_box in sp_results[nuscenes_sample_token]:
            if int(sample_box['tracking_id']) in referred_box_ids:
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
for nuprompt_sample_token in pt_results.keys():
    if nuprompt_sample_token not in nuprompt_results:
        print(nuprompt_sample_token)
    else:
        nuprompt_results_small[nuprompt_sample_token] = nuprompt_results[nuprompt_sample_token]

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

# For use in the NuPrompt codebase evaluation scrip.
with open(f'output/sm_predictions/{exp_name}/StreamPETR_NuPrompt_results_{exp_name}_small.json', 'w') as file:
    json.dump(nuprompt_format_small, file, indent=4)

with open(f'output/sm_predictions/{exp_name}/StreamPETR_NuPrompt_results_{exp_name}.json', 'w') as file:
    json.dump(nuprompt_format, file, indent=4)