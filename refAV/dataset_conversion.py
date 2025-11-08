"""
Processes the scenario mining annotation files downloaded from RefAV. 
Also processes object tracking prediction files in the format Argoverse2 submission format.
"""

from pathlib import Path
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import pyarrow.feather as feather
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import os
import os
import pandas as pd
from pathlib import Path

from av2.utils.io import read_feather, read_city_SE3_ego
from av2.evaluation.tracking.utils import save
from av2.structures.cuboid import CuboidList
from refAV.paths import *
from refAV.utils import get_ego_SE3, get_log_split


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
    # TODO: revert back from NuPrompt experiments
    exclude_columns = ['log_id']#, 'prompt', 'mining_category']
    
    # Process each log_id
    for log_id in tqdm(unique_log_ids):
        # Create directory for this log_id
        log_dir = base_dir / str(log_id)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all entries for this log_id
        log_data = df[df['log_id'] == log_id]
        
        #log_prompts = log_data['prompt'].unique()
        #log_data = log_data[log_data['prompt'] == log_prompts[0]]
        
        # Keep only the "others" columns
        filtered_data = log_data.drop(columns=exclude_columns)
        
        # Save to a feather file
        output_path = log_dir / 'city_SE3_egovehicle.feather'
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
    #Filtering can dramatically speed up evaluation

    id_stats = {}
    kept_ids = []
    kept_ids_per_timestamp = {}

    for frame in track_data:
        timestamp = frame['timestamp_ns']
        ids = frame['track_id']
        scores = frame['score']
        categories = frame['label']

        kept_ids_per_timestamp[timestamp] = []

        for index, id in enumerate(ids):

            if id not in id_stats:
                #Track length, min confidence
                id_stats[id] = ([timestamp], scores[index], categories[index])
            else:
                #Weight longer tracks higher
                #print(id_stats[id])
                id_stats[id] = (id_stats[id][0] + [timestamp], scores[index] + id_stats[id][1], categories[index])

    id_stats_by_category = {}
    for id, stats in id_stats.items():
        id_timestamps, score, category = stats
        
        if category not in id_stats_by_category:
            id_stats_by_category[category] = []
        id_stats_by_category[category].append((id, score))

    for category, category_ids in id_stats_by_category.items():
        sorted_ids = sorted(category_ids, key=lambda row:row[1], reverse=True)

        if category in ['REGULAR_VEHCILE', 'PEDESTRIAN', 'BOLLARD', 'CONSTRUCTION_CONE', 'CONSTRUCTION_BARREL']:
            topk = 200
        else:
            topk = 100
        for i in range(min(topk, len(sorted_ids))):
            #print(sorted_ids[i][1])
            kept_ids.append(sorted_ids[i][0])
            id_timestamps = id_stats[sorted_ids[i][0]][0]
            for timestamp in id_timestamps:
                kept_ids_per_timestamp[timestamp].append(sorted_ids[i][0])

        #Make sure that all frames have at least one kept id
        for timestamp in kept_ids_per_timestamp.keys():
            while len(kept_ids_per_timestamp[timestamp]) == 0 and i < len(sorted_ids):
                if timestamp in id_stats[sorted_ids[i][0]][0]:
                    kept_ids.append(sorted_ids[i][0])
                    for id_timestamp in id_stats[sorted_ids[i][0]][0]:
                        kept_ids_per_timestamp[id_timestamp].append(sorted_ids[i][0])
                i += 1

    #print(f'filtered from {len(id_stats.keys())} to {len(kept_ids)} ids')

    return kept_ids

def process_sequences(log_id, track_data, dataset_dir, base_output_dir, filter=True):
    ego_poses = read_city_SE3_ego(dataset_dir / log_id)
    rows = []

    if filter and 'score' in track_data[0]:
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
            if filter and track_ids[i] not in kept_ids:
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
        #print(f"Created feather file: {output_path}")

        add_ego_to_annotation(output_path.parent, output_path.parent)
        output_path.unlink()

    #print(f'Tracking predictions processed for log-id {log_id}')



def pickle_to_feather(dataset_dir, input_pickle_path, base_output_dir="output"):
    """
    Convert pickle file to feather files with the specified format.
    Creates a separate feather file for each log_id in its own directory.
    
    Args:
        input_pickle_path: Path to the input pickle file
        base_output_dir: Base directory for output folders
    """

    #print(f"Reading pickle file: {input_pickle_path}")

    with open(input_pickle_path, 'rb') as f:
        data = pickle.load(f)

    with mp.Pool(max(1, int(.9*(os.cpu_count())))) as pool:
        pool.starmap(process_sequences, 
                    [(log_id, track_data, dataset_dir, base_output_dir)
                    for log_id, track_data in data.items()])



def add_ego_to_annotation(log_dir:Path, output_dir:Path=Path('output')):

    split = get_log_split(log_dir)
    annotations_df = read_feather(log_dir / 'annotations.feather')
    ego_df = read_feather(AV2_DATA_DIR / split / log_dir.name / 'city_SE3_egovehicle.feather')
    ego_df['log_id'] = log_dir.name
    ego_df['track_uuid'] = 'ego'
    ego_df['category'] = 'EGO_VEHICLE'
    ego_df['length_m'] = 4.877
    ego_df['width_m'] = 2
    ego_df['height_m'] = 1.473
    ego_df['qw'] = 1
    ego_df['qx'] = 0
    ego_df['qy'] = 0
    ego_df['qz'] = 0
    ego_df['tx_m'] = 0
    ego_df['ty_m'] = 0
    ego_df['tz_m'] = 0
    ego_df['num_interior_pts'] = 1

    if 'score' in annotations_df.columns:
        ego_df['score'] = 1

    synchronized_timestamps = annotations_df['timestamp_ns'].unique()
    ego_df = ego_df[ego_df['timestamp_ns'].isin(synchronized_timestamps)]

    combined_df = pd.concat([annotations_df, ego_df], ignore_index=True)
    feather.write_feather(combined_df, output_dir / 'sm_annotations.feather')
    #print(f'Successfully added ego to annotations for log {log_dir.name}.')


def feather_to_csv(feather_path, output_dir):
    df:pd.DataFrame = feather.read_feather(feather_path)
    output_filename = output_dir + '/output.csv'
    df.to_csv('output.csv', index=False)
    #print(f'Successfully saved to {output_filename}')


def mining_category_from_df(df:pd.DataFrame, mining_category:str):
    
    log_timestamps = np.sort(df['timestamp_ns'].unique())
    category_df = df[df['mining_category'] == mining_category]

    #Keys timestamps, values list of track_uuids
    category_objects = {}

    for timestamp in log_timestamps:
        timestamp_uuids = category_df[category_df['timestamp_ns'] == timestamp]['track_uuid'].unique()
        category_objects[timestamp] = list(timestamp_uuids)

    return category_objects
    

def process_log_prompt(log_id, prompt, sm_annotations, output_dir, SM_DATA_DIR, read_feather, mining_category_from_df, get_ego_SE3, CuboidList, Rotation, save):
    """Process a single log_id and prompt combination."""
    log_df = sm_annotations[sm_annotations['log_id'] == log_id]
    lpp_df = log_df[log_df['prompt'] == prompt]
    
    frames = []
    
    split = get_log_split(Path(log_id))
    log_dir = SM_DATA_DIR / split / log_id 
    (output_dir / log_id).mkdir(exist_ok=True)
    
    annotations = read_feather(log_dir / 'sm_annotations.feather')
    log_timestamps = np.sort(annotations['timestamp_ns'].unique())
    all_uuids = list(annotations['track_uuid'].unique())
    ego_poses = get_ego_SE3(log_dir)
    
    referred_objects = mining_category_from_df(lpp_df, 'REFERRED_OBJECT')
    related_objects = mining_category_from_df(lpp_df, 'RELATED_OBJECT')
    
    for timestamp in log_timestamps:
        frame = {}
        timestamp_annotations = annotations[annotations['timestamp_ns'] == timestamp]
        
        timestamp_uuids = list(timestamp_annotations['track_uuid'].unique())
        ego_to_city = ego_poses[timestamp]
        
        frame['seq_id'] = (log_id, prompt)
        frame['timestamp_ns'] = timestamp
        frame['ego_translation_m'] = list(ego_to_city.translation)
        frame['description'] = prompt
        
        n = len(timestamp_uuids)
        frame['translation_m'] = np.zeros((n, 3))
        frame['size'] = np.zeros((n,3), dtype=np.float32)
        frame['yaw'] = np.zeros(n, dtype=np.float32)
        frame['velocity_m_per_s'] = np.zeros((n,3))
        frame['label'] = np.zeros(n, dtype=np.int32)
        frame['name'] = np.zeros(n, dtype='<U31')
        frame['track_id'] = np.zeros(n, dtype=np.int32)
        
        for i, track_uuid in enumerate(timestamp_uuids):
            track_df = timestamp_annotations[timestamp_annotations['track_uuid'] == track_uuid]
            cuboid = CuboidList.from_dataframe(track_df)[0]
            
            if track_df.empty:
                continue
                
            ego_coords = track_df[['tx_m', 'ty_m', 'tz_m']].to_numpy()
            size = track_df[['length_m', 'width_m', 'height_m']].to_numpy()
            translation_m = ego_to_city.transform_from(ego_coords)
            yaw = Rotation.from_matrix(ego_to_city.compose(cuboid.dst_SE3_object).rotation).as_euler('zxy')[0]
            
            if timestamp in referred_objects and track_uuid in referred_objects[timestamp]:
                category = "REFERRED_OBJECT"
                label = 0
            elif timestamp in related_objects and track_uuid in related_objects[timestamp]:
                category = "RELATED_OBJECT"
                label = 1
            else:
                category = "OTHER_OBJECT"
                label = 2
                
            frame['translation_m'][i,:] = translation_m
            frame['size'][i,:] = size
            frame['yaw'][i] = yaw
            frame['velocity_m_per_s'][i,:] = np.zeros(3)
            frame['label'][i] = label
            frame['name'][i] = category
            frame['track_id'][i] = all_uuids.index(track_uuid)
            
        frames.append(frame)
        
    EVALUATION_SAMPLING_FREQUENCY = 5   
    frames = frames[::EVALUATION_SAMPLING_FREQUENCY]
    
    sequences = {(log_id, prompt): frames}
    
    output_path = output_dir / log_id / f'{prompt}_{log_id[:8]}_ref_gt.pkl'
    save(sequences, output_path)
    return f'Scenario pkl file for {prompt}_{log_id[:8]} saved successfully.'

def create_gt_mining_pkls_parallel(scenario_mining_annotations_path, output_dir: Path, num_processes=None):
    """
    Generates both a pkl file for evaluation in parallel.
    
    Args:
        scenario_mining_annotations_path: Path to annotations
        output_dir: Path to output directory
        num_processes: Number of CPU cores to use (None = use all available)
    """
    
    sm_annotations = read_feather(scenario_mining_annotations_path)
    log_ids = sm_annotations['log_id'].unique()
    
    # Create a list of (log_id, prompt) tuples to process
    tasks = []
    for log_id in log_ids:
        log_df = sm_annotations[sm_annotations['log_id'] == log_id]
        prompts = log_df['prompt'].unique()
        for prompt in prompts:
            tasks.append((log_id, prompt))
    
    # If num_processes is not specified, use all available cores
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # The number of processes shouldn't exceed the number of tasks
    num_processes = min(num_processes, len(tasks))
    
    # Create output directories first to avoid race conditions
    for log_id in log_ids:
        (output_dir / log_id).mkdir(exist_ok=True)
    
    # Define the worker function with partial to fix most parameters
    worker_func = partial(
        process_log_prompt,
        sm_annotations=sm_annotations,
        output_dir=output_dir,
        SM_DATA_DIR=SM_DATA_DIR,  # Make sure this is defined in the global scope
        read_feather=read_feather,
        mining_category_from_df=mining_category_from_df,
        get_ego_SE3=get_ego_SE3,
        CuboidList=CuboidList,
        Rotation=Rotation,
        save=save
    )
    
    # Display a progress bar for the overall process
    #print(f"Processing {len(tasks)} log-prompt combinations using {num_processes} processes")
    
    # Use a Pool of workers to process in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use imap to get results as they complete and to allow for a progress bar
        results = list(tqdm(
            pool.starmap(worker_func, tasks),
            total=len(tasks),
            desc="Processing log-prompt pairs"
        ))
    
    # Print summary
    print(f"Completed processing {len(results)} log-prompt combinations")
    return results

if __name__ == "__main__":
    tracking_val_predictions = Path('tracker_predictions/Le3DE2E_tracking_predictions_val.pkl')
    sm_val_feather = Path('av2_sm_downloads/scenario_mining_val_annotations.feather')

    # TODO: revert
    streamPETR_preds_feather = Path('/home/crdavids/Trinity-Sync/StreamPETR/ego_poses.feather')
    streamPETR_dir = Path('/home/crdavids/Trinity-Sync/refbot/output/tracker_predictions/StreamPETR_Tracking/nuprompt_val_large')

    separate_scenario_mining_annotations(streamPETR_preds_feather, streamPETR_dir)
    #pickle_to_feather(AV2_DATA_DIR, tracking_val_predictions, SM_PRED_DIR)
    #create_gt_mining_pkls_parallel(sm_val_feather, SM_DATA_DIR, num_processes=max(1, int(.5*os.cpu_count())))





