from refAV.eval import combine_pkls, evaluate_pkls, compile_results
import numpy as np
from pathlib import Path
from refAV.utils import create_mining_pkl, get_log_split, read_feather
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
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
import time # Added for potential debugging or timing
from av2.datasets.sensor.splits import TRAIN, TEST, VAL

with open('/home/crdavids/Trinity-Sync/refbot/baselines/groundingSAM/log_id_to_start_index.json') as file:
    log_id_to_start_index = json.load(file)


#create eval timestamps
eval_timestamps = {}
dataloader = SensorDataloader(paths.AV2_DATA_DIR, with_annotations=False)

for split in [TEST, VAL]:
    for log_id in tqdm(split):
        log_split = get_log_split(log_id)
        sm_data_dir = Path('/data3/crdavids/refAV/dataset')
        df = read_feather(sm_data_dir / log_split / log_id / 'annotations_with_ego.feather')
        df_timestamps = sorted(df['timestamp_ns'].unique())[::5]
        log_eval_timestamps = [int(timestamp) for timestamp in df_timestamps]
        eval_timestamps[log_id] = log_eval_timestamps
        print(len(log_eval_timestamps))
        
with open('output/eval_timestamps.json', 'w') as file:
    json.dump(eval_timestamps, file, indent=4)

"""
all_descriptions = []
with open('/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/log_prompt_pairs_test.json', 'rb') as file:
    lpp_test = json.load(file)
with open('/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/log_prompt_pairs_val.json', 'rb') as file:
    lpp_val = json.load(file)
for lpp in [lpp_test, lpp_val]:
    for prompts in lpp.values():
        all_descriptions.extend(prompts)

all_descriptions = list(set(all_descriptions))
code = "objects=get_objects_of_prompt(log_dir, description)\noutput_scenario(objects,description,log_dir,output_dir)"

for description in all_descriptions:
    output_path = paths.LLM_PRED_DIR / 'objects_of_assigned_description' / (description+'.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as file:
        file.write(code)
"""
