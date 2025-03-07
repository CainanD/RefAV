from pathlib import Path
from av2.utils.io import read_feather, read_city_SE3_ego
from av2.evaluation.tracking.utils import save, load
from tqdm import tqdm
import pyarrow.feather as feather
import pandas as pd
import shutil
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation
import os
import json
import glob
import re
import uuid
from eval import combine_matching_pkls
from av2.datasets.sensor.splits import TEST, TRAIN, VAL
from paths import EGO_VEHICLE_UUIDS
import os
import pandas as pd
import argparse
from pathlib import Path
import random



def separate_scenario_mining_annotations(input_feather_path, base_annotation_dir):
    """
    Converts a feather file containing log data into individual feather files.
    
    Each log_id gets its own directory, and one description per log is randomly sampled.
    The log_id, description, and mining_category columns are excluded from the output.
    
    Args:
        input_feather_path (str): Path to the input feather file
        base_annotation_dir (str): Base directory where output folders will be created
    """
    # Create base directory if it doesn't exist
    base_dir = Path(base_annotation_dir)
    base_dir.mkdir(exist_ok=True, parents=True)
    
    # Read the input feather file
    print(f"Reading input feather file: {input_feather_path}")
    df = pd.read_feather(input_feather_path)
    
    # Get unique log_ids
    unique_log_ids = df['log_id'].unique()
    print(f"Found {len(unique_log_ids)} unique log IDs")
    
    # Columns to exclude
    exclude_columns = ['log_id', 'description', 'mining_category']
    
    # Process each log_id
    for log_id in unique_log_ids:
        # Create directory for this log_id
        log_dir = base_dir / str(log_id)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all entries for this log_id
        log_data = df[df['log_id'] == log_id]
        

        log_prompts = log_data['prompt'].unique()
        log_data = log_data[log_data['prompt'] == log_prompts[0]]
        
        # Keep only the "others" columns
        filtered_data = log_data.drop(columns=exclude_columns)
        
        # Save to a feather file
        output_path = log_dir / 'sm_annotations.feather'
        filtered_data.to_feather(output_path)
        print(f"Saved {output_path}")
    
    print(f"Conversion complete. Files saved to {base_annotation_dir}")


def euler_to_quaternion(yaw, pitch=0, roll=0):
    """
    Convert Euler angles to quaternion.
    Assuming we only have yaw and want to convert to qw, qx, qy, qz
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

def filter_ids_by_score(track_data):
    id_stats = {}
    kept_ids = []

    for frame in track_data:
        ids = frame['track_id']
        scores = frame['score']

        for index, id in enumerate(ids):
            if id not in id_stats:
                #Track length, min confidence
                id_stats[id] = (1, scores[index])
            else:
                id_stats[id] = (id_stats[id][0]+1, min(scores[index], id_stats[id][1]))

    for id, stats in id_stats.items():
        track_length, min_confidence = stats

        if (track_length == 1 and min_confidence > .05) or (track_length > 1 and min_confidence > .005):
            kept_ids.append(id)

    print(f'filtered from {len(id_stats.keys())} to {len(kept_ids)} ids')

    return kept_ids


def pickle_to_feather(dataset_dir, input_pickle_path, base_output_dir="output"):
    """
    Convert pickle file to feather files with the specified format.
    Creates a separate feather file for each log_id in its own directory.
    
    Args:
        input_pickle_path: Path to the input pickle file
        base_output_dir: Base directory for output folders
    """
    dataset_dir = Path(dataset_dir)
    base_output_dir = Path(base_output_dir)
    
    print(f"Reading pickle file: {input_pickle_path}")

    with open(input_pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Process each log_id
    for log_id, track_data in data.items():
        ego_poses = read_city_SE3_ego(dataset_dir / log_id)

        rows = []

        if 'score' in track_data[0]:
            kept_ids = filter_ids_by_score(track_data)
        
        for frame in track_data:
            timestamp = frame['timestamp_ns']

            # Extract size dimensions
            lengths = frame['size'][:, 0]  # First column for length
            widths = frame['size'][:, 1]   # Second column for width
            heights = frame['size'][:, 2]   # Third column for height
            
            # Get translations
            city_to_ego = ego_poses[timestamp].inverse()

            city_coords = frame['translation_m']
            ego_coords = city_to_ego.transform_from(city_coords)
            
            # Get yaws
            yaws = frame['yaw']
            
            # Get categories (names)
            categories = frame['name']
            
            # Get track IDs
            track_ids = frame['track_id']
            

            if 'score' in frame:
                scores = frame['score']
            else:
                kept_ids = track_ids

            
            # Process each object in the frame
            ego_yaws = np.zeros(len(track_ids))
            for i in range(len(track_ids)):
                if track_ids[i] not in kept_ids:
                    continue

                city_rotation = Rotation.from_euler('xyz', [0, 0, yaws[i]]).as_matrix()
                
                ego_yaws[i] = Rotation.from_matrix(city_to_ego.rotation @ city_rotation).as_euler('zxy')[0]
                # Convert yaw to quaternion
                qw, qx, qy, qz = euler_to_quaternion(ego_yaws[i])

                if 'score' in frame:
                    row = {
                        'timestamp_ns': timestamp,
                        'track_uuid': str(track_ids[i]),  # Convert track ID to string
                        'category': categories[i],
                        'length_m': lengths[i],
                        'width_m': widths[i],
                        'height_m': heights[i],
                        'qw': qw,
                        'qx': qx,
                        'qy': qy,
                        'qz': qz,
                        'tx_m': ego_coords[i, 0],
                        'ty_m': ego_coords[i, 1],
                        'tz_m': ego_coords[i, 2],
                        'num_interior_pts': 0,  # Default value as this info isn't in the original data
                        'score': scores[i]
                    }
                else:
                    row = {
                        'timestamp_ns': timestamp,
                        'track_uuid': str(track_ids[i]),  # Convert track ID to string
                        'category': categories[i],
                        'length_m': lengths[i],
                        'width_m': widths[i],
                        'height_m': heights[i],
                        'qw': qw,
                        'qx': qx,
                        'qy': qy,
                        'qz': qz,
                        'tx_m': ego_coords[i, 0],
                        'ty_m': ego_coords[i, 1],
                        'tz_m': ego_coords[i, 2],
                        'num_interior_pts': 0  # Default value as this info isn't in the original data
                    }
                rows.append(row)
        
        if rows:
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Ensure columns are in the correct order
            columns = [
                'timestamp_ns', 'track_uuid', 'category', 
                'length_m', 'width_m', 'height_m',
                'qw', 'qx', 'qy', 'qz',
                'tx_m', 'ty_m', 'tz_m',
                'num_interior_pts', 'score'
            ]
            df = df[columns]
            
            # Create directory structure and save feather file
            log_dir = base_output_dir / str(log_id)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = log_dir / "annotations.feather"
            df.to_feather(output_path)
            print(f"Created feather file: {output_path}")


if __name__ == "__main__":

    feather_file = '/home/crdavids/Downloads/av2_annotations/av2_annotations/test_annotations.feather'
    log_parent_dir = Path('/home/crdavids/Trinity-Sync/av2-api/output/dataset/test')
    output_path = '/home/crdavids/Trinity-Sync/av2-api/output/dataset/test/0c6e62d7-bdfa-3061-8d3d-03b13aa21f68'
    tracker_predictions_path = Path('/home/crdavids/Trinity-Sync/av2-api/output/misc/predpkls')
    tracker_predictions_dir = Path('/home/crdavids/Trinity-Sync/av2-api/output/tracker_predictions/val')
    trinity_dataset_dir = Path('/data3/shared/datasets/ArgoVerse2/Sensor/val')
    dataset_base_dir = Path('/data3/crdavids/refAV/dataset/val')
    dataset_dir = Path(f'/data3/crdavids/refAV/dataset/val')
    gt_pickle_dir=  Path('/home/crdavids/Trinity-Sync/av2-api/output/misc/gtpkls')
    #convert_seq_ids('/home/crdavids/Trinity-Sync/av2-api/output/tracker_predictions/test', backup=False)
    #create_scenario_mining_dataset_feather(Path('/home/crdavids/Trinity-Sync/av2-api/output/eval/test/combined_gt.pkl'), dataset_dir)
    #create_refAV_annotations(trinity_dataset_dir, dataset_base_dir)
    #pickle_to_feather(dataset_dir, tracker_predictions_path, tracker_predictions_dir)
    #dataset_annotations_to_log_annotations(feather_file, log_parent_dir)
    #create_refAV_annotations(dataset_dir, tracker_predictions_dir)
    #combine_and_enrich_feather_files(gt_pickle_dir, dataset_base_dir, 'output/misc/scenario_mining_val_annotations.feather')
    combine_matching_pkls(gt_pickle_dir, tracker_predictions_dir, Path('/home/crdavids/Trinity-Sync/av2-api/output/misc'))
    #feather_to_csv('/home/crdavids/Trinity-Sync/av2-api/output/eval/val/scenario_mining_val_annotations.feather','/home/crdavids/Trinity-Sync/av2-api/output/eval/val')




