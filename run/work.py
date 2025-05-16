from refAV.eval import combine_pkls, evaluate_pkls
import numpy as np
from pathlib import Path
from refAV.utils import create_mining_pkl
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


original_paths = []
for pkl in list(paths.TRACKER_DOWNLOAD_DIR.iterdir()):
    original_paths.append(pkl)

for pkl in original_paths:
    if '.pkl' not in pkl.name or 'Detections' not in pkl.name:
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

        if sequence_id not in sub_sampled_pkl:
            sub_sampled_pkl[sequence_id] = []
        for frame in frames:
            if frame['timestamp_ns'] in eval_timestamps_by_log_id[log_id]:
                sub_sampled_pkl[sequence_id].append(frame)

    sub_sampled_pkl_path = paths.TRACKER_DOWNLOAD_DIR / (pkl.stem + '_2hz.pkl')
    with open(sub_sampled_pkl_path, 'wb') as file:
        pickle.dump(sub_sampled_pkl, file)

    pickle_to_feather(paths.AV2_DATA_DIR / split, sub_sampled_pkl_path, paths.TRACKER_PRED_DIR / (tracker + '_2hz') / split)
