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


def generate_ego_vehicle_uuids():

    if EGO_VEHICLE_UUIDS.exists():
        print('Ego vehicle UUIDs already exist!')
        return
    
    ego_vehicle_uuid = {}
    for split in [TEST, TRAIN, VAL]:
        for log_id in split:
            ego_vehicle_uuid[log_id] = str(uuid.uuid4())
            print(f'Ego vehicle UUID {ego_vehicle_uuid[log_id]} generated for log {log_id}')

    with open(EGO_VEHICLE_UUIDS, 'w') as f:
        json.dump(ego_vehicle_uuid, f, indent=4)


def add_ego_to_annotation(log_dir:Path, output_dir:Path=Path('output'), is_gt=True):

    if is_gt:
        with open(EGO_VEHICLE_UUIDS, 'r') as f:
            ego_vehicle_uuids = json.load(f)
        ego_uuid = ego_vehicle_uuids[log_dir.name]
    else:
        ego_uuid = 'ego'

    annotations_df = read_feather(log_dir / 'annotations.feather')
    ego_df = read_feather(log_dir / 'city_SE3_egovehicle.feather')
    ego_df['track_uuid'] = ego_uuid
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
    feather.write_feather(combined_df, output_dir / 'annotations_with_ego.feather')
    print(f'Successfully added ego to annotations for log {log_dir.name}.')

def feather_to_csv(feather_path, output_dir):
    df:pd.DataFrame = feather.read_feather(feather_path)
    output_filename = output_dir + '/output.csv'
    df.to_csv('output.csv', index=False)
    print(f'Successfully saved to {output_filename}')


def combine_and_enrich_feather_files(scenario_mining_path, av2_dataset_path, output_file_path):
    """
    Combines all scenario mining annotation feather files into a single file and enriches them
    with additional columns from corresponding av2_dataset annotations.
    
    Args:
        scenario_mining_path (str): Path to the scenario_mining_dataset folder
        av2_dataset_path (str): Path to the av2_dataset folder
        output_file_path (str): Path where the combined feather file should be saved
    """
    # Convert paths to Path objects
    scenario_mining_path = Path(scenario_mining_path)
    av2_dataset_path = Path(av2_dataset_path)
    output_file_path = Path(output_file_path)
    
    # Create output directory if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize an empty list to store all dataframes
    all_mining_dfs = []
    
    # Get all log_id subdirectories in the scenario_mining_dataset folder
    log_dirs = [d for d in scenario_mining_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(log_dirs)} log directories to process")
    
    # Iterate through each log_id subdirectory
    for log_dir in tqdm(log_dirs, desc="Processing log directories"):
        log_id = log_dir.name
        
        # Get all feather files in the current log directory that have a corresponding *_ref_gt.pkl file
        all_feather_files = list(log_dir.glob("*_annotations.feather"))
        feather_files = []
        
        for feather_file in all_feather_files:
            # Extract the base name without the _annotations.feather suffix
            base_name = feather_file.name.replace("_annotations.feather", "")
            # Check if there's a corresponding *_ref_gt.pkl file
            ref_gt_file = log_dir / f"{base_name}_ref_gt.pkl"
            if ref_gt_file.exists():
                feather_files.append(feather_file)
            else:
                print(f"Skipping {feather_file} as it has no corresponding {ref_gt_file.name}")
        
        if not feather_files:
            print(f"No annotation feather files found in {log_dir}")
            continue
        
        # Read and combine all feather files in the current log directory
        for feather_file in feather_files:
            try:
                mining_df = pd.read_feather(feather_file)
                all_mining_dfs.append(mining_df)
            except Exception as e:
                print(f"Error reading {feather_file}: {e}")
    
    # Combine all scenario mining dataframes
    if not all_mining_dfs:
        raise ValueError("No valid scenario mining data found")
    
    combined_mining_df = pd.concat(all_mining_dfs, ignore_index=True)
    print(f"Combined {len(all_mining_dfs)} scenario mining files with {len(combined_mining_df)} total rows")
    
    # Get the unique columns we want to keep from the scenario mining data
    mining_columns = ['log_id', 'prompt', 'track_uuid', 'mining_category', 'timestamp_ns']
    
    # Process the av2_dataset files to add additional columns
    enriched_rows = []
    
    # Track logs that don't exist in av2_dataset
    missing_logs = set()
    
    # Iterate through unique log_ids in the combined mining data
    unique_log_ids = combined_mining_df['log_id'].unique()
    
    for log_id in tqdm(unique_log_ids, desc="Enriching with av2 data"):
        av2_log_path = av2_dataset_path / log_id / "annotations_with_ego.feather"
        
        if not av2_log_path.exists():
            missing_logs.add(log_id)
            continue
        
        # Read the av2 annotations for this log_id
        try:
            av2_df = pd.read_feather(av2_log_path)
            
            # Get mining data for this log_id
            log_mining_df = combined_mining_df[combined_mining_df['log_id'] == log_id]
            
            # Create a multi-index for faster lookups
            av2_df_indexed = av2_df.set_index(['track_uuid', 'timestamp_ns'])
            
            # Process each row in the mining data for this log
            for _, mining_row in log_mining_df.iterrows():
                track_uuid = mining_row['track_uuid']
                timestamp_ns = mining_row['timestamp_ns']
                
                # Try to find the corresponding row in av2 data
                try:
                    av2_row = av2_df_indexed.loc[(track_uuid, timestamp_ns)].iloc[0] if isinstance(
                        av2_df_indexed.loc[(track_uuid, timestamp_ns)], pd.DataFrame) else av2_df_indexed.loc[(track_uuid, timestamp_ns)]
                    
                    # Create a combined row with all columns
                    combined_row = mining_row.to_dict()
                    
                    # Add av2 columns that aren't in mining data
                    for col in av2_row.index:
                        if col not in mining_columns and col != 'num_interior_points':
                            combined_row[col] = av2_row[col]
                    
                    enriched_rows.append(combined_row)
                except (KeyError, IndexError):
                    # If no matching av2 data, just keep the original mining row
                    enriched_rows.append(mining_row.to_dict())
        
        except Exception as e:
            print(f"Error processing av2 data for {log_id}: {e}")
    
    if missing_logs:
        print(f"Warning: {len(missing_logs)} logs not found in av2_dataset: {', '.join(list(missing_logs)[:5])}{'...' if len(missing_logs) > 5 else ''}")
    
    # Create the final enriched dataframe
    if not enriched_rows:
        print("Warning: No enriched data rows created. Saving only combined mining data.")
        final_df = combined_mining_df
    else:
        final_df = pd.DataFrame(enriched_rows)
        print(f"Created enriched dataframe with {len(final_df)} rows and {len(final_df.columns)} columns")
    
    # Save the final dataframe as a feather file
    final_df.to_feather(output_file_path)
    print(f"Saved combined and enriched data to {output_file_path}")
    
    return final_df


def create_refAV_annotations(av2_dataset_dir, target_base_dir=None):
    """
    Copy city_SE3_egovehicle.feather files and map directory contents from source to target directories.
    
    Args:
        av2_dataset_dir: Base directory containing source log_id subdirectories
        target_base_dir: Base directory containing target log_id subdirectories
    """
    if target_base_dir == None:
        target_base_dir = av2_dataset_dir

    source_base = Path(av2_dataset_dir)
    target_base = Path(target_base_dir)
    
    # Get all log_id subdirectories from source
    source_log_dirs = [d for d in source_base.iterdir() if d.is_dir()]
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    print(f"Found {len(source_log_dirs)} log directories in source")
    
    for source_log_dir in source_log_dirs:
        log_id = source_log_dir.name
        
        # Paths for feather file
        source_file = source_log_dir / "city_SE3_egovehicle.feather"
        target_log_dir = target_base / log_id
        target_file = target_log_dir / "city_SE3_egovehicle.feather"
        
        # Paths for map directory
        source_map_dir = source_log_dir / "map"
        target_map_dir = target_log_dir / "map"
        

        # Check if target directory exists
        target_log_dir.mkdir(exist_ok=True)
        if not target_log_dir.exists():
            print(f"Warning: Target directory not found for {log_id}")
            skipped_count += 1
            continue
            
        # Handle feather file copy
        if source_file.exists():
            add_ego_to_annotation(source_log_dir, target_log_dir)
            shutil.copy2(source_file, target_file)
            print(f"Copied feather file for {log_id}")

        else:
            print(f"Warning: Feather file not found in {log_id}")
            
        """    
        # Handle map directory copy
        if source_map_dir.exists():
            # Remove existing target map directory if it exists
            if target_map_dir.exists():
                shutil.rmtree(target_map_dir)
                
            # Copy the entire map directory and its contents
            shutil.copytree(source_map_dir, target_map_dir)
            print(f"Copied map directory for {log_id}")
            copied_count += 1
        else:
            print(f"Warning: Map directory not found in {log_id}")
            skipped_count += 1
        """
                
            
    # Print summary
    print("\nSummary:")
    print(f"Total directories processed: {len(source_log_dirs)}")
    print(f"Successfully copied: {copied_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")

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

def load_pickle(file_path):
    """Load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, file_path):
    """Save data to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

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
    data = load_pickle(input_pickle_path)
    
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


def pickle_to_text(input_pickle_path, output_text_path=None, max_tracks=None, max_timestamps=None):
    """
    Convert a pickle file containing track data to a readable text file.
    
    Args:
        input_pickle_path: Path to the input pickle file
        output_text_path: Path for the output text file (if None, generates automatic name)
        max_tracks: Maximum number of tracks to process (if None, processes all)
        max_timestamps: Maximum number of timestamps per track to process (if None, processes all)
    """
    # Generate output filename if not provided
    if output_text_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_text_path = f"track_data_{timestamp}.txt"
    
    print(f"Reading pickle file: {input_pickle_path}")
    data = load_pickle(input_pickle_path)
    
    print(f"Writing to text file: {output_text_path}")
    with open(output_text_path, "w") as f:
        # Write header info
        f.write(f"Track Data Export\n")
        f.write(f"Generated on: {datetime.now()}\n")
        f.write(f"Total number of tracks: {len(data)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Process tracks
        tracks_to_process = list(data.items())[:max_tracks] if max_tracks else list(data.items())
        
        for log_id, track_data in tracks_to_process:
            f.write(f"Log ID: {log_id}\n")
            f.write("=" * 80 + "\n")
            
            timestamps_to_process = track_data[:max_timestamps] if max_timestamps else track_data
            
            for timestamp_data in timestamps_to_process:
                f.write(f"\nTimestamp: {timestamp_data['timestamp_ns']}\n")
                f.write("-" * 40 + "\n")
                
                # Write translation data
                f.write("Translation (x, y, z):\n")
                for point in timestamp_data['translation_m']:
                    f.write(f"    {point}\n")
                
                # Write size data
                f.write("\nSize (length, width, height):\n")
                for size in timestamp_data['size']:
                    f.write(f"    {size}\n")
                
                # Write yaw data
                f.write(f"\nYaw: {timestamp_data['yaw']}\n")
                
                # Write velocity data
                f.write("\nVelocity (x, y, z):\n")
                for vel in timestamp_data['velocity_m_per_s']:
                    f.write(f"    {vel}\n")
                
                # Write label and name
                f.write(f"\nLabels: {timestamp_data['label']}\n")
                f.write(f"Names: {timestamp_data['name']}\n")
                
                # Write scores and track IDs
                f.write(f"Track IDs: {timestamp_data['track_id']}\n")
                
                f.write("\n" + "-" * 80 + "\n")
            
            f.write("\n\n")

    print(f"Conversion complete! Output written to: {output_text_path}")


def dataset_annotations_to_log_annotations(dataset_feather, log_parent_dir:Path):
    
    dataset = feather.read_feather(dataset_feather)
    log_ids = dataset['log_id'].unique()

    for log_id in log_ids:
        log_df = dataset[dataset['log_id'] == log_id]
        log_dir = log_parent_dir / log_id
        log_dir.mkdir(exist_ok=True)

        feather.write_feather(log_df, log_dir/'annotations.feather')
        print(f'Successfully created annotations for log {log_id}.')


def convert_seq_ids(input_dir, backup=True):
    """
    Convert sequence IDs from string format to tuple format in all pickle files.
    
    Args:
        input_dir: Directory containing pickle files
        backup: Whether to create backup files
    """
    # Find all pickle files recursively
    pickle_files = glob.glob(os.path.join(input_dir, "**/*.pkl"), recursive=True)
    
    if not pickle_files:
        print(f"No pickle files found in {input_dir}")
        return
    
    print(f"Found {len(pickle_files)} pickle files to process")
    
    for file_path in pickle_files:
        print(f"Processing {file_path}")
        
        try:
            # Load the pickle file
            data = load_pickle(file_path)
            
            if backup:
                # Create a backup
                backup_path = f"{file_path}.bak"
                save_pickle(data, backup_path)
                print(f"Created backup at {backup_path}")
            
            # Check if the data structure has the expected format (dictionary of sequences)
            if not isinstance(data, dict):
                print(f"  Warning: Unexpected format in {file_path}, skipping")
                continue
            
            # Create a new dictionary with converted keys
            new_data = {}
            modified = False
            
            # Process the top-level keys (sequence IDs)
            for seq_id, frames in data.items():
                if isinstance(seq_id, str) and "_" in seq_id:
                    # Parse the string seq_id
                    # Try to extract log_dir.name and description from the string
                    match = re.match(r"([^_]+)_(.+)", seq_id)
                    if match:
                        description, log_name = match.groups()
                        new_seq_id = (log_name, description)
                        new_data[new_seq_id] = frames
                        modified = True
                        print(f"  Converting key: {seq_id} -> {new_seq_id}")
                    else:
                        new_data[seq_id] = frames
                        print(f"  Could not parse seq_id: {seq_id}")
                else:
                    new_data[seq_id] = frames
            
            # Process the nested seq_id values in each frame
            for seq_id, frames in new_data.items():
                for frame in frames:
                    if 'seq_id' in frame and isinstance(frame['seq_id'], str) and "_" in frame['seq_id']:
                        # Parse the string seq_id
                        match = re.match(r"([^_]+)_(.+)", frame['seq_id'])
                        if match:
                            description, log_name = match.groups()
                            frame['seq_id'] = (log_name, description)
                            modified = True
                            print(f"  Converting nested seq_id: {frame['seq_id']}")
            
            # Save the modified data if changes were made
            if modified:
                save_pickle(new_data, file_path)
                print(f"  Updated {file_path}")
            else:
                print(f"  No changes needed for {file_path}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print("Conversion completed")

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




