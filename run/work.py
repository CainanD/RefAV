from refAV.eval import combine_pkls, evaluate_pkls
import numpy as np
from pathlib import Path
from refAV.utils import create_mining_pkl, get_ego_SE3, print_indented_dict
import pickle
import refAV.paths as paths
import yaml
import json
import shutil
from tqdm import tqdm
from refAV.dataset_conversion import pickle_to_feather
import pickle
import os
import concurrent.futures
from tqdm import tqdm
from refAV.dataset_conversion import pickle_to_feather
import time # Added for potential debugging or timing
from av2.evaluation.tracking.eval import evaluate
import argparse


#parser = argparse.ArgumentParser(description="Example script with arguments")
#parser.add_argument("--tracker", type=str, required=True)
#args = parser.parse_args()
#tracker=args.tracker
tracker = 'LT3D'
with open('/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/annotations_testsplit.pkl', 'rb') as file:
    labels = pickle.load(file)

with open(paths.TRACKER_DOWNLOAD_DIR / (tracker + '_Detections_test_2hz.pkl'), 'rb') as file:
    predictions = pickle.load(file)

objective_metric = 'HOTA'
max_range_m = 50
dataset_dir = paths.AV2_DATA_DIR / 'test'

metrics = evaluate(predictions, labels, objective_metric, max_range_m, dataset_dir, out=f'misc/{tracker}')
print(metrics)

"""
def create_default_frame(log_dir, timestamp)->dict:


    frame = {}
    ego_poses = get_ego_SE3(log_dir)
    ego_to_city = ego_poses[timestamp]

    frame['seq_id'] = log_id
    frame['timestamp_ns'] = timestamp
    frame['ego_translation_m'] = list(ego_to_city.translation)

    frame['translation_m'] = np.zeros((1, 3))
    frame['size'] = np.zeros((1,3), dtype=np.float32)
    frame['yaw'] = np.zeros(1, dtype=np.float32)
    frame['label'] = np.zeros(1, dtype=np.int32)
    frame['name'] = np.zeros(1, dtype='<U31')
    frame['track_id'] = np.zeros(1, dtype=np.int32)
    frame['score'] = np.zeros(1, dtype=np.float32)
    frame['name'][0] = "OTHER_OBJECT"
    frame['label'][0] = 2

    return frame


original_paths = []
for pkl in list(paths.TRACKER_DOWNLOAD_DIR.iterdir()):
    original_paths.append(pkl)

for pkl in original_paths:
    if '.pkl' not in pkl.name or 'Detections' not in pkl.name or '_2hz' in pkl.name:
        continue
    
    tracker = pkl.stem.split('_')[0] + '_' + pkl.stem.split('_')[1]
    split = pkl.stem.split('_')[2]

    with open(paths.SM_DOWNLOAD_DIR / 'eval_timestamps.json', 'rb') as file:
        eval_timestamps_by_log_id = json.load(file)

    with open(pkl, 'rb') as file:
        sequences = pickle.load(file)

    sub_sampled_pkl = {}
    for sequence_id, frames in sequences.items():
        log_id = sequence_id
        log_timestamps = eval_timestamps_by_log_id[log_id]

        if sequence_id not in sub_sampled_pkl:
            sub_sampled_pkl[sequence_id] = []
        for frame in frames:
            if frame['timestamp_ns'] in eval_timestamps_by_log_id[log_id]:
                sub_sampled_pkl[sequence_id].append(frame)
                log_timestamps.remove(frame['timestamp_ns'])

        if len(log_timestamps) > 0:
            for timestamp in log_timestamps:
                sub_sampled_pkl[sequence_id].append(create_default_frame(log_id, timestamp))
                print(f'{log_id}: {log_timestamps}')

    sub_sampled_pkl_path = paths.TRACKER_DOWNLOAD_DIR / (pkl.stem + '_2hz.pkl')
    with open(sub_sampled_pkl_path, 'wb') as file:
        pickle.dump(sub_sampled_pkl, file)

    pickle_to_feather(paths.AV2_DATA_DIR / split, sub_sampled_pkl_path, paths.TRACKER_PRED_DIR / (tracker + '_2hz') / split)
"""