import json
import numpy as np
from collections import defaultdict
import warnings
import regex as re
import matplotlib
import pandas as pd
matplotlib.use('Agg') # Use a non-interactive backend for saving plots to files
import matplotlib.pyplot as plt
from pathlib import Path
from refAV.utils import get_log_split, swap_keys_and_listed_values
import refAV.paths as paths

def convert_detections_to_tracker(log_prompt_pairs_path, detections_dir:Path):

    with open(log_prompt_pairs_path, 'rb') as file:
        lpp = json.load(file)

    num_found = 0
    print(len(list(detections_dir.iterdir())))
    for log_id, prompts in lpp.items():
        split = get_log_split(log_id)
        log_df = None

        for prompt in prompts:
            file_found = False
            for detection_file in detections_dir.iterdir():
                if not detection_file.is_file():
                    continue

                substrings = detection_file.stem.split('_')
                if len(substrings) < 3 or substrings[1] != log_id:
                    continue
                
                description_substrings = substrings[2:]
                reconstructed_description = description_substrings[0]
                for i in range(1, len(description_substrings)):
                    reconstructed_description += ("_" + description_substrings[i])

                safe_prompt = re.sub(r'[^\w\-]+', '_', prompt).strip('_').lower()[:50]

                if reconstructed_description == safe_prompt and '.feather' in detection_file.name:
                    num_found += 1
                    file_found=True
                    prompt_df = pd.read_feather(detection_file)
                    prompt_df['prompt'] = prompt
                    prompt_df['category'] = "REFERRED_OBJECT"

                    if log_df is None:
                        log_df = prompt_df
                    else:
                        log_df = pd.concat((log_df, prompt_df))
                    break
            if not file_found:
                print(safe_prompt)
                print(reconstructed_description)
                raise Exception(f'{prompt} detection file not found for log {log_id}')

        dest = paths.TRACKER_PRED_DIR / 'groundingSAM' / split / log_id / 'sm_annotations.feather'
        dest.parent.mkdir(exist_ok=True, parents=True)
        log_df.to_feather(dest)

if __name__ == "__main__":
    print('Converting to tracker ...')
    convert_detections_to_tracker('/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/log_prompt_pairs_test.json',
                                Path('baselines/groundingSAM/output'))
