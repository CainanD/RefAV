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


def eval_similarity_scores(
    input_json_path: str, 
    output_json_path: str, 
    majority_threshold: float = 0.75,
):
    """
    Processes a JSON file of tracker data, clusters confidence scores, selects tracks,
    and optionally visualizes the clusters.

    Args:
        input_json_path (str): Path to the input JSON file.
        output_json_path (str): Path to save the output JSON file.
        k (int): The number of top clusters to select. If k <= 0, it defaults to 1.
        majority_threshold (float, optional): Min proportion of a track's scores in top k clusters. Defaults to 0.5.
        plot_output_dir (str, optional): Directory to save cluster visualizations. If None, no plots are saved.
    """
    output_json_path = Path(output_json_path)
    input_json_path = Path(input_json_path)
    log_id_feather = {}

    # Load input JSON
    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_json_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_json_path}'.")
        return

    if not isinstance(data, dict):
        print(f"Error: Input JSON root must be a dictionary (object). Found type: {type(data)}.")
        return

    output_data = defaultdict(lambda: defaultdict(list))

    for prompt_id, logs in data.items():
        if not isinstance(logs, dict):
            warnings.warn(
                f"Warning: Content for prompt '{prompt_id}' is not a dictionary of logs. Skipping.",
                UserWarning
            )
            output_data[prompt_id] = {}
            continue

        all_confidence_scores_for_prompt = []
        track_confidences_map = {}

        if not logs:
            output_data[prompt_id] = {}
            continue

        for log_id, tracks_in_log in logs.items():
            #split = get_log_split(log_id)
            #if log_id not in log_id_feather:
            #    log_id_feather[log_id] = pd.read_feather(
            #        paths.TRACKER_PRED_DIR / input_json_path.stem / split / log_id / 'sm_annotations.feather')

            output_data[prompt_id][log_id] = []
            if not isinstance(tracks_in_log, dict):
                warnings.warn(
                    f"Warning: Content for log '{log_id}' (prompt '{prompt_id}') is not a dict of tracks. Skipping.",
                    UserWarning
                )
                continue
            
            for track_uuid, timestamps in tracks_in_log.items():
                current_track_scores = []
                track_confidences_map[(log_id, track_uuid)] = []
                if not isinstance(timestamps, dict):
                    warnings.warn(
                        f"Warning: Timestamps for track '{track_uuid}' (log '{log_id}', prompt '{prompt_id}') "
                        f"is not a dictionary. Skipping scores for this track.",
                        UserWarning
                    )
                    continue
                
                for _, confidence in timestamps.items():
                    if isinstance(confidence, (int, float)):
                        all_confidence_scores_for_prompt.append(confidence)
                        current_track_scores.append(confidence)
                    else:
                        warnings.warn(
                            f"Warning: Invalid confidence value '{confidence}' (type: {type(confidence)}) "
                            f"for prompt '{prompt_id}', log '{log_id}', track '{track_uuid}'. Skipping.",
                            UserWarning
                        )
                track_confidences_map[(log_id, track_uuid)] = current_track_scores
        
        scores_np = np.array(all_confidence_scores_for_prompt)
        #Set the threshold at the 75%-ile 
        scores_np = np.sort(scores_np)
        threshold = scores_np[int(majority_threshold*(len(scores_np)))]
        for (log_id, track_uuid), track_scores in track_confidences_map.items():
            
            #track_pred_df = log_id_feather[log_id]
            #category = track_pred_df[track_pred_df['track_uuid'] == track_uuid]['category'].unique()[0]

            if (np.sum(np.where(track_scores > threshold, 1, 0))/len(track_scores)) > 0.5:
                output_data[prompt_id][log_id].append(track_uuid)

        output_file = output_json_path / input_json_path.stem / f'{prompt_id}.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data[prompt_id], f, indent=4)

    return output_data

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
