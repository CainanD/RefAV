import json
from pathlib import Path
from scipy.spatial.transform import Rotation
import pandas as pd
from tqdm import tqdm
import numpy as np
import shutil
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from refAV.paths import NUSCENES_DIR, NUSCENES_AV2_DATA_DIR

NUSCENES_VAL_LOG_IDS = \
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

NUSC_CATEGORY_TO_AV2_CATEGORY = {
    "vehicle.motorcycle": "MOTORCYCLE",
    "vehicle.bicycle": "BICYCLE",
    "vehicle.bus.bendy": "BUS",
    "vehicle.trailer": "TRAILER",
    "vehicle.car": "CAR",
    "human.pedestrian.adult": "PEDESTRIAN",
    "human.pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.wheelchair": "PEDESTRIAN",
    "human.pedestrian.stroller": "PEDESTRIAN",
    "human.pedestrian.personal_mobility": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "vehicle.truck": "TRUCK",
    }


# Only used for visualization
#output_path = Path('output/tracker_predictions/nuscenes_ground_truth/val')

def separate_scenario_mining_annotations(input_df, base_annotation_dir, filename):
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
    
    df = input_df#pd.read_csv(input_df)

    # Get unique log_ids
    unique_log_ids = df['log_id'].unique()
    print(f"Found {len(unique_log_ids)} unique log IDs")
    
    exclude_columns = ['log_id']
    
    # Process each log_id
    for log_id in tqdm(unique_log_ids):
        # Create directory for this log_id
        log_dir = base_dir / str(log_id)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Get all entries for this log_id
        log_data = df[df['log_id'] == log_id]
    
        filtered_data:pd.DataFrame = log_data.drop(columns=exclude_columns).reset_index(drop=True)
        
        # Save to a feather file
        output_file_dir = log_dir 
        output_file_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_file_dir/filename
        if 'track_uuid' in filtered_data.columns:
            filtered_data['track_uuid'] = filtered_data['track_uuid'].astype(str)
        filtered_data.to_feather(output_file)
        print(f"Saved {output_file}")
    
    print(f"Conversion complete. Files saved to {base_annotation_dir}")


def token(obj):
    #Convert list of dicts with 'token' key to dict indexed by token
    if isinstance(obj, dict) and 'sample_token' in obj:
        # Check if it's a list of dicts with 'token' keys
        return (obj['sample_token'], obj)
    if isinstance(obj, dict) and 'token' in obj:
        # Check if it's a list of dicts with 'token' keys
        return (obj['token'], obj)
    return obj
 

def process_tracking_predictions(batch_args):
    batch_data, global_data, output_path = batch_args

    sample = global_data['sample']
    sample_data = global_data['sample_data']
    ego_pose = global_data['ego_pose']
    calibrated_sensor = global_data['calibrated_sensor']
    sensor = global_data['sensor']

    # Initialize local dataframes for this batch
    local_df = []
    local_pose_df = []
    local_intrinsics_df = []
    local_calibration_df = []

    for sample_token, tracks in batch_data:

        scene_token = sample[sample_token]['scene_token']
        timestamp = 1000*int(sample[sample_token]['timestamp']) # convert micro to nano

        for sample_dict in sample_data[sample_token]:
            pose = ego_pose[sample_dict['ego_pose_token']]
            entry = {
            'log_id':scene_token,
            'timestamp_ns':1000*int(pose['timestamp']),
            'qw':pose['rotation'][0],
            'qx':pose['rotation'][1],
            'qy':pose['rotation'][2],
            'qz':pose['rotation'][3],
            'tx_m':pose['translation'][0],
            'ty_m':pose['translation'][1],
            'tz_m':pose['translation'][2]}
            local_pose_df.append(entry)


            sensor_calibration = calibrated_sensor[sample_dict['calibrated_sensor_token']]
            sensor_q = sensor_calibration['rotation']
            sensor_t = sensor_calibration['translation']
            
            sensor_name = sensor[sensor_calibration['sensor_token']]['channel']
            if 'CAM' in sensor_name:
                filename = sample_dict['filename']
                output_dir = output_path/scene_token/'sensors'/'cameras'/sensor_name
                output_dir.mkdir(exist_ok=True, parents=True)
                output_filename:Path = output_dir/f'{timestamp}.jpg'

                if not output_filename.exists():
                    shutil.copy2(nuscenes_path.parent/filename, output_dir/f'{timestamp}.jpg')

                intrinsics = sensor_calibration['camera_intrinsic']

                intrinsics_entry = {
                    'log_id':scene_token,
                    'sensor_name':sensor_name,
                    'fx_px':intrinsics[0][0],
                    'fy_px':intrinsics[1][1],
                    'cx_px':intrinsics[0][2],
                    'cy_px':intrinsics[1][2],
                    'height_px':900,
                    'width_px':1600
                }
                extrisics_entry = {
                    'log_id':scene_token,
                    'sensor_name':sensor_name,
                    'qw':sensor_q[0],
                    'qx':sensor_q[1],
                    'qy':sensor_q[2],
                    'qz':sensor_q[3],
                    'tx_m':sensor_t[0],
                    'ty_m':sensor_t[1],
                    'tz_m':sensor_t[2]
                }
                local_intrinsics_df.append(intrinsics_entry)
                local_calibration_df.append(extrisics_entry)

        #Add ego cuboid
        entry = {
        'log_id':scene_token,
        'timestamp_ns':timestamp,
        'track_uuid':0,
        'category':"EGO_VEHICLE",
        'length_m':4.877,
        'width_m':2.0,
        'height_m':1.473,
        'qw':1,
        'qx':0,
        'qy':0,
        'qz':0,
        'tx_m':1.422,
        'ty_m':0,
        'tz_m':0.25,
        'score':1.0}
        local_df.append(entry)

        for track in tracks:
            
            q_wxyz = track['rotation']
            q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
            object_to_world_r = Rotation.from_quat(q_xyzw)
            object_to_world_t = np.array(track['translation'])

            q_wxyz = pose['rotation']
            q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
            ego_to_world_r = Rotation.from_quat(q_xyzw)
            world_to_ego_r = ego_to_world_r.inv()
            ego_to_world_t = np.array(pose['translation'])

            # object_to_ego = world_to_ego @ object_to_world
            # = world_to_ego_r@((object_to_world_r@object)+object_to_world_t)+world_to_ego_t
            # assuming that object is located at 0,0,0
            # object_t = ego_to_world_r^-1@(object_to_world_t)-ego_to_world_t
            object_to_ego_t = world_to_ego_r.as_matrix()@(object_to_world_t-ego_to_world_t)
            object_to_ego_r = object_to_world_r*world_to_ego_r
            quat = object_to_ego_r.as_quat()

            # Add row using dictionary
            entry = {
            'log_id':scene_token,
            'timestamp_ns':timestamp,
            'track_uuid':track['tracking_id'],
            'category':track['tracking_name'].upper(),
            'length_m':track['size'][1],
            'width_m':track['size'][0],
            'height_m':track['size'][2],
            'qw':quat[3],
            'qx':quat[0],
            'qy':quat[1],
            'qz':quat[2],
            'tx_m':object_to_ego_t[0],
            'ty_m':object_to_ego_t[1],
            'tz_m':object_to_ego_t[2],
            'score':track['tracking_score']}
            local_df.append(entry)

    return {
        'df': local_df,
        'pose_df': local_pose_df,
        'intrinsics_df': local_intrinsics_df,
        'calibration_df': local_calibration_df,
    }


def process_nuscenes_logs(batch_args):
    """
    Arguements
        batch_args: list[(annotation_tokens, nuscenes_data)]
    
    Returns
        Dict of dataframes in the AV2 format
    """

    annotation_tokens, nuscenes_data, val, nusc_category_to_tracking_category_map, output_path = batch_args

    sample = nuscenes_data['sample']
    sample_data = nuscenes_data['sample_data']
    ego_pose = nuscenes_data['ego_pose']
    calibrated_sensor = nuscenes_data['calibrated_sensor']
    sensor = nuscenes_data['sensor']
    scene = nuscenes_data['scene']
    instance = nuscenes_data['instance']
    category = nuscenes_data['category']
    sample_annotation = nuscenes_data['sample_annotation']

    # Initialize local dataframes for this batch
    local_df = []
    local_pose_df = []
    local_intrinsics_df = []
    local_calibration_df = []

    encountered_samples = []
    for annotation_token in annotation_tokens:

        sample_token = sample_annotation[annotation_token]['sample_token']
        scene_token = sample[sample_token]['scene_token']

        timestamp = 1000*int(sample[sample_token]['timestamp']) # convert micro to nano
        track = sample_annotation[annotation_token]

        if sample_token not in encountered_samples:
            encountered_samples.append(sample_token)
            for sample_dict in sample_data[sample_token]:
                if not sample_dict['is_key_frame']:
                    continue
                
                pose = ego_pose[sample_dict['ego_pose_token']]
                entry = {
                'log_id':scene_token,
                'timestamp_ns':1000*int(pose['timestamp']),
                'qw':pose['rotation'][0],
                'qx':pose['rotation'][1],
                'qy':pose['rotation'][2],
                'qz':pose['rotation'][3],
                'tx_m':pose['translation'][0],
                'ty_m':pose['translation'][1],
                'tz_m':pose['translation'][2]}
                local_pose_df.append(entry)


                sensor_calibration = calibrated_sensor[sample_dict['calibrated_sensor_token']]
                sensor_q = sensor_calibration['rotation']
                sensor_t = sensor_calibration['translation']
                
                sensor_name = sensor[sensor_calibration['sensor_token']]['channel']
                if 'CAM' in sensor_name:
                    filename = sample_dict['filename']
                    output_dir = output_path/scene_token/'sensors'/'cameras'/sensor_name
                    output_dir.mkdir(exist_ok=True, parents=True)
                    output_filename:Path = output_dir/f'{timestamp}.jpg'

                    if not output_filename.exists():
                        shutil.copy2(nuscenes_path.parent/filename, output_dir/f'{timestamp}.jpg')

                    intrinsics = sensor_calibration['camera_intrinsic']

                    intrinsics_entry = {
                        'log_id':scene_token,
                        'sensor_name':sensor_name,
                        'fx_px':intrinsics[0][0],
                        'fy_px':intrinsics[1][1],
                        'cx_px':intrinsics[0][2],
                        'cy_px':intrinsics[1][2],
                        'height_px':900,
                        'width_px':1600
                    }
                    extrisics_entry = {
                        'log_id':scene_token,
                        'sensor_name':sensor_name,
                        'qw':sensor_q[0],
                        'qx':sensor_q[1],
                        'qy':sensor_q[2],
                        'qz':sensor_q[3],
                        'tx_m':sensor_t[0],
                        'ty_m':sensor_t[1],
                        'tz_m':sensor_t[2]
                    }
                    local_intrinsics_df.append(intrinsics_entry)
                    local_calibration_df.append(extrisics_entry)

            #Add ego cuboid
            entry = {
            'log_id':scene_token,
            'timestamp_ns':timestamp,
            'track_uuid':0,
            'category':"EGO_VEHICLE",
            'length_m':4.877,
            'width_m':2.0,
            'height_m':1.473,
            'qw':1,
            'qx':0,
            'qy':0,
            'qz':0,
            'tx_m':1.422,
            'ty_m':0,
            'tz_m':0.25,
            'score':1.0}
            local_df.append(entry)

        pose = ego_pose[sample_data[sample_token]['ego_pose_token']]
        q_wxyz = track['rotation']
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        object_to_world_r = Rotation.from_quat(q_xyzw)
        object_to_world_t = np.array(track['translation'])

        q_wxyz = pose['rotation']
        q_xyzw = [q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]
        ego_to_world_r = Rotation.from_quat(q_xyzw)
        world_to_ego_r = ego_to_world_r.inv()
        ego_to_world_t = np.array(pose['translation'])

        # object_to_ego = world_to_ego @ object_to_world
        # = world_to_ego_r@((object_to_world_r@object)+object_to_world_t)+world_to_ego_t
        # assuming that object is located at 0,0,0
        # object_t = ego_to_world_r^-1@(object_to_world_t)-ego_to_world_t
        object_to_ego_t = world_to_ego_r.as_matrix()@(object_to_world_t-ego_to_world_t)
        object_to_ego_r = object_to_world_r*world_to_ego_r
        quat = object_to_ego_r.as_quat()

        nusc_category = category[instance[track['instance_token']]['category_token']]


        # Add row using dictionary
        entry = {
        'log_id':scene_token,
        'timestamp_ns':timestamp,
        'track_uuid':track['instance_token'],
        'category':nusc_category_to_tracking_category_map[nusc_category],
        'length_m':track['size'][1],
        'width_m':track['size'][0],
        'height_m':track['size'][2],
        'qw':quat[3],
        'qx':quat[0],
        'qy':quat[1],
        'qz':quat[2],
        'tx_m':object_to_ego_t[0],
        'ty_m':object_to_ego_t[1],
        'tz_m':object_to_ego_t[2]}
        local_df.append(entry)

    return {
        'df': local_df,
        'pose_df': local_pose_df,
        'intrinsics_df': local_intrinsics_df,
        'calibration_df': local_calibration_df,
    }

def nuscenes_to_av2():

    num_processes = min(cpu_count()-1, 64)  # Limit to avoid memory issues
    annotation_tokens = list(sample_annotation_local.keys())

    batch_args = []

    batched_tokens = []
    for annotation_token in annotation_tokens:
        sample_token = sample_annotation_local[annotation_token]['sample_token']
        scene_token = sample_local[sample_token]['scene_token']

        scene_name = scene_local[scene_token]['name']
        if scene_name not in val:
            continue

        if len(batched_tokens) <= len(annotation_tokens)/num_processes:
            batched_tokens.append(annotation_token)
        else:
            batch_args.append((deepcopy(batched_tokens), nusc_data, val, nusc_category_to_tracking_category_map))
            batched_tokens = []

    print(f"Processing {len(annotation_tokens)} samples in {len(batch_args)} batches using {num_processes} processes...")

    all_results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        batch_futures = [executor.submit(process_nuscenes_logs, args) for args in batch_args]
        
        for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc="Processing batches"):
            result = future.result()
            all_results.append(result)


    # Combine all results
    print("Combining results...")
    all_df = []
    all_pose_df = []
    all_intrinsics_df = []
    all_calibration_df = []

    for result in all_results:
        all_df.extend(result['df'])
        all_pose_df.extend(result['pose_df'])
        all_intrinsics_df.extend(result['intrinsics_df'])
        all_calibration_df.extend(result['calibration_df'])

    # Convert to DataFrames
    df = pd.DataFrame(all_df)
    pose_df = pd.DataFrame(all_pose_df)
    intrinsics_df = pd.DataFrame(all_intrinsics_df)
    calibration_df = pd.DataFrame(all_calibration_df)

    # Remove duplicates from intrinsics and calibration (based on log_id and sensor_name)
    intrinsics_df = intrinsics_df.drop_duplicates(subset=['log_id', 'sensor_name'])
    calibration_df = calibration_df.drop_duplicates(subset=['log_id', 'sensor_name'])


    df.to_csv('nuscenes_gt.csv')
    pose_df.to_csv('ego_poses.csv', index=False)
    calibration_df.to_csv('ego_SE3_sensor.csv')
    intrinsics_df.to_csv('intrinsics.csv')

    separate_scenario_mining_annotations(df, output_path, 'sm_annotations.feather')
    separate_scenario_mining_annotations(pose_df, output_path, 'city_SE3_egovehicle.feather')
    separate_scenario_mining_annotations(calibration_df, output_path, 'egovehicle_SE3_sensor.feather')
    separate_scenario_mining_annotations(intrinsics_df, output_path, 'intrinsics.feather')


def nuscenes_tracking_to_av2_tracking(tracking_results_path:Path, output_path:Path):

    with open(tracking_results_path, 'rb') as file:
        tracking_results = json.load(file)['results']   

    sm_annotations_path = output_path / 'nuscenes_tracking.csv' 
    intrinsics_path = output_path / 'intrinsics.csv'
    pose_path = output_path / 'ego_SE3_sensor.csv'
    calibration_path = output_path / 'ego_poses.csv'

    if sm_annotations_path.exists() and intrinsics_path.exists() and pose_path.exists() and calibration_path.exists():
        df = pd.read_csv(sm_annotations_path)
        pose_df = pd.read_csv(pose_path)
        intrinsics_df = pd.read_csv(intrinsics_path)
        calibration_df = pd.read_csv(calibration_path)
    else:

        tracking_items = list(tracking_results.items())
        num_processes = min(cpu_count(), 32)  # Limit to avoid memory issues
        batch_size = max(1, len(tracking_items) // num_processes)
        batches = [tracking_items[i:i + batch_size] for i in range(0, len(tracking_items), batch_size)]
        print(f"Processing {len(tracking_items)} samples in {len(batches)} batches using {num_processes} processes...")

        batch_args = [(batch, nusc_data, output_path) for batch in batches]

        all_results = []
        with ProcessPoolExecutor(max_workers=num_processes+1) as executor:
            batch_futures = [executor.submit(process_tracking_predictions, args) for args in batch_args]
            
            for future in tqdm(as_completed(batch_futures), total=len(batch_futures), desc="Processing batches"):
                result = future.result()
                all_results.append(result)

        # Combine all results
        print("Combining results...")
        all_df = []
        all_pose_df = []
        all_intrinsics_df = []
        all_calibration_df = []

        for result in all_results:
            all_df.extend(result['df'])
            all_pose_df.extend(result['pose_df'])
            all_intrinsics_df.extend(result['intrinsics_df'])
            all_calibration_df.extend(result['calibration_df'])

        # Convert to DataFrames
        df = pd.DataFrame(all_df)
        pose_df = pd.DataFrame(all_pose_df)
        intrinsics_df = pd.DataFrame(all_intrinsics_df)
        calibration_df = pd.DataFrame(all_calibration_df)

        # Remove duplicates from intrinsics and calibration (based on log_id and sensor_name)
        intrinsics_df = intrinsics_df.drop_duplicates(subset=['log_id', 'sensor_name'])
        calibration_df = calibration_df.drop_duplicates(subset=['log_id', 'sensor_name'])

        df.to_csv(output_path/'nuscenes_tracking.csv')
        pose_df.to_csv(output_path/'ego_poses.csv', index=False)
        calibration_df.to_csv(output_path/'ego_SE3_sensor.csv')
        intrinsics_df.to_csv(output_path/'intrinsics.csv')

    separate_scenario_mining_annotations(df, output_path, 'sm_annotations.feather')
    separate_scenario_mining_annotations(pose_df, output_path, 'city_SE3_egovehicle.feather')
    separate_scenario_mining_annotations(calibration_df, output_path, 'egovehicle_SE3_sensor.feather')
    separate_scenario_mining_annotations(intrinsics_df, output_path, 'intrinsics.feather')


# TODO: Put NuPrompt data in RefAV visualizer

if __name__ == "__main__":

    with open(nuscenes_path/'sample.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        sample_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'calibrated_sensor.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        calibrated_sensor_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'sensor.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        sensor_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'scene.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        scene_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'ego_pose.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        ego_pose_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'sample_data.json', 'rb') as file:
        data_list = json.load(file)
        sample_data_local = {}
        for data in tqdm(data_list):
            sample_token = data['sample_token']
            if sample_token not in sample_data_local:
                sample_data_local[sample_token] = []
            sample_data_local[sample_token].append(data)
    with open(nuscenes_path/'instance.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        instance_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'category.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        category_local = {key: value for (key, value) in data_list}
    with open(nuscenes_path/'sample_annotation.json', 'rb') as file:
        data_list = json.load(file, object_hook=token)
        sample_annotation_local = {key: value for (key, value) in data_list}

    nusc_data = {
        "sample":sample_local,
        "calibrated_sensor":calibrated_sensor_local,
        "sensor":sensor_local,
        "scene":scene_local,
        "ego_pose":ego_pose_local,
        "sample_data":sample_data_local,
        "sample_annotation":sample_annotation_local,
        "instance":instance_local,
        "category":category_local
    }
    
    nuscenes_tracking_to_av2_tracking(
        tracking_results_path=Path('/home/crdavids/Trinity-Sync/PF-Track/ckpts/PF-Track-Models/f3_fullres_all/results/results_nusc_tracking.json'), 
        output_path=Path('output/tracker_predictions/PFTrack_Tracking/nuprompt_val')
    )

    # Only used for ground truth visualization
    #nuscenes_to_av2(Path('output/tracker_predictions/nuscenes_ground_truth/nuprompt_val'))
