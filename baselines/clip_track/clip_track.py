import torch
import os
import clip
import numpy as np
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import argparse
import shutil

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.io import read_feather
from av2.structures.cuboid import CuboidList
from av2.datasets.sensor.constants import RingCameras
from av2.geometry.camera.pinhole_camera import PinholeCamera
import refAV.paths as paths
from refAV.utils import print_indented_dict, swap_keys_and_listed_values

def convert_numpy_types(data):
    """
    Recursively converts NumPy types within a dictionary (and nested structures)
    to their corresponding Python types. Handles common data structures like
    lists, tuples, and nested dictionaries.

    Args:
        data: The input data, which can be a dictionary, list, tuple, or a
            single value. NumPy types are assumed to be within this data
            structure.

    Returns:
        A new data structure with all NumPy types converted to their
        corresponding Python types. Returns the original data if no
        NumPy types are found or if the input is not a dict, list, or tuple.
    """
    if isinstance(data, dict):
        # Iterate through the dictionary and recursively convert values.
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        # Iterate through the list or tuple and recursively convert elements.
        return type(data)(convert_numpy_types(item) for item in data)
    elif isinstance(data, np.int64):
        # Base case: convert NumPy type to Python type.
        return int(data)  # Use the .item() method for conversion
    elif isinstance(data, np.float64):
        # Base case: convert NumPy type to Python type.
        return float(data)  # Use the .item() method for conversion
    else:
        # If it's not a NumPy type, return the original data.
        return data

def get_track_features(log_dir, data_loader:AV2SensorDataLoader, clip_model, preprocessor, device) -> torch.tensor:
    #Keys are track_uuids, values are clip features

    track_features = {}

    log_id = log_dir.name
    split = log_dir.parent.name
    tracker = log_dir.parent.parent.name

    df = read_feather(log_dir / 'sm_annotations.feather')
    log_timestamps = sorted(df['timestamp_ns'].unique())
    if len(log_timestamps) > 50:
        log_timestamps = log_timestamps[::5]

    track_uuids = sorted(df['track_uuid'].unique())

    track_features_path = Path(f'baselines/clip_track/track_features/{tracker}/{split}/{log_id}/track_features_dict.pt')
    if track_features_path.exists():
        track_features = torch.load(track_features_path, map_location=device, weights_only=False)
        return track_features
    else:
        track_features_path.parent.mkdir(parents=True, exist_ok=True)

    for track_uuid in tqdm(track_uuids):

        image_features_dict = {}
        track_df = df[df["track_uuid"] == track_uuid]
        track_timestamps = sorted(track_df['timestamp_ns'].unique())

        for timestamp in track_timestamps:
            timestamp_df = track_df[track_df['timestamp_ns'] == timestamp]
            cuboid = CuboidList.from_dataframe(timestamp_df)[0]
            bbox_vertices = cuboid.vertices_m

            for camera in RingCameras:
                cam_name = camera.value
                uv, points_cam, is_valid = data_loader.project_ego_to_img_motion_compensated(bbox_vertices, cam_name, timestamp, timestamp, log_id)
                if np.any(is_valid):
                    try:
                        img_path = data_loader.get_closest_img_fpath(log_id, cam_name, timestamp)
                        img = Image.open(img_path)
                        W = img.width
                        H = img.height

                        uv = uv[is_valid]
                        
                        #Bypasses the edge case where two points along the same x or y value are the only two valid points
                        x_min = np.clip(np.min(uv[:,0])-30, 0, W-1)
                        x_max = np.clip(np.max(uv[:,0])+30, 0, W-1)
                        y_min = np.clip(np.min(uv[:,1])-30, 0, H-1)
                        y_max = np.clip(np.max(uv[:,1])+30, 0, H-1)
                        box = (x_min, y_min, x_max, y_max)

                        crop = img.crop(box)
                        #Path(f'baselines/sample_crops/{tracker}/{log_id}').mkdir(parents=True, exist_ok=True)
                        #crop.save(f'baselines/sample_crops/{tracker}/{log_id}/{cuboid.category}_{track_uuid}.png')

                        with torch.no_grad():
                            crop = preprocessor(crop).unsqueeze(0).to(device)
                            image_features_dict[timestamp] = clip_model.encode_image(crop)
                    except Exception as e: 
                        pass

        if image_features_dict:
            track_features[track_uuid] = image_features_dict

    torch.save(track_features, track_features_path)
    return track_features

def get_description_features(description, clip_model, device) -> torch.tensor:

    description_features_path = Path(f'baselines/clip_track/description_features/{description}.pt')
    if description_features_path.exists():
            return torch.load(description_features_path, map_location=device, weights_only=False)
    else:
        description_features_path.parent.mkdir(parents=True, exist_ok=True)

    text_inputs = clip.tokenize(description).to(device)
    text_features = clip_model.encode_text(text_inputs)
    torch.save(text_features, description_features_path)

    return text_features


def eval_prompt(prompt, log_ids, tracker, split, load_model, device):

    print(f"Worker process {os.getpid()} is using device {device}")

    if load_model:
        model, preprocess = clip.load("ViT-L/14", device=device)
        data_loader = AV2SensorDataLoader(data_dir=paths.AV2_DATA_DIR / split, labels_dir=paths.AV2_DATA_DIR / split)
    else:
        model = None
        preprocess = None
        data_loader = None
        device = 'cpu'

    clip_scores = {}
    clip_scores[prompt] = {}
    description_features = get_description_features(prompt, model, device)
    #print(description_features)
    
    np.random.shuffle(log_ids)
    for log_id in log_ids:
        log_dir = paths.TRACKER_PRED_DIR / tracker / split / log_id
        track_features = get_track_features(log_dir, data_loader, model, preprocess, device)
        for track_uuid, timestamped_features in track_features.items():
            for timestamp, image_features in timestamped_features.items():

                cos_similarity = torch.cosine_similarity(description_features.flatten(), image_features.flatten(), dim=0)

                if log_id not in clip_scores[prompt]:
                    clip_scores[prompt][log_id] = {}
                if track_uuid not in clip_scores[prompt][log_id]:
                    clip_scores[prompt][log_id][track_uuid] = {}
                clip_scores[prompt][log_id][track_uuid][timestamp] = cos_similarity.item()

    return clip_scores


def eval_similarity_scores(tracker_json, k):

    #Given a json file with structure {prompt:{log_id:{track_uuid:{timestamp:confidence},},},}
    # we want to assign a yes/no per track

    #For each prompt, we have all of the tracks in each associated log.
    #Each track is split into timestamps and given a confidence value for each timestamp.
    #For each prompt,
    # 1. Lay out the confidence scores along a number line
    # 2. Cluster the confidence scores and select the the top k clusters
    # 3. If the majority of the track applies to the topk clusters, store that track_id
    # 4. Add visualization for the number line.

    # output a json file with the structure {prompt:{log_id:[track_uuid,],},}

    pass


if __name__ == '__main__':

    import multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument("--tracker", type=str, required=True, help="Enter the name of the tracker from the tracker_downloads folder.")
    parser.add_argument("--split", type=str, default='test', help="Enter the split to evaluate")
    parser.add_argument("--start_index", type=int, default=0, help="Enter the name of the experiment from experiments.yml you would like to run.")
    parser.add_argument("--end_index", type=int, default=500, help="Enter the name of the experiment from experiments.yml you would like to run.")
    parser.add_argument("--load_model", type=bool, default=True, help="Enter the name of the experiment from experiments.yml you would like to run.")

    args = parser.parse_args()

    # Define device early, but don't load CUDA objects until in the worker or after spawn
    main_process_device = "cuda" if torch.cuda.is_available() else "cpu"

    log_prompt_pairs = paths.SM_DOWNLOAD_DIR / f'log_prompt_pairs_{args.split}.json'
    with open(log_prompt_pairs, 'rb') as file:
        lpp = json.load(file)

    plp = swap_keys_and_listed_values(lpp)

    mini_plp = {}
    for prompt in list(plp.keys())[args.start_index:args.end_index]:
        mini_plp[prompt] = plp[prompt]

    # Now pass necessary arguments to the worker function
    # The worker function will load the model and dataloader itself
    with Pool(processes=os.cpu_count()//10) as pool:
        # Pass args needed to initialize dataloader and model in worker
        clip_scores = pool.starmap(eval_prompt, 
                [(prompt, log_ids, args.tracker, args.split, args.load_model, main_process_device) 
                 for i, (prompt, log_ids) in enumerate(plp.items())])

    all_clip_scores = {}
    for clip_score in clip_scores:
        all_clip_scores.update(clip_score)

    all_clip_scores = convert_numpy_types(all_clip_scores)

    # Ensure you are writing to the file, not opening with 'rb'
    with open(f'baselines/clip_track/similarity_scores/{args.tracker}_{args.start_index}_{args.end_index}.json', 'w') as file:
        json.dump(all_clip_scores, file, indent=4) 