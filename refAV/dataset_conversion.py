"""
Processes the scenario mining annotation files downloaded from RefAV.
Also processes object tracking prediction files in the format Argoverse2 submission format.
"""
import json
import os
import pickle
import multiprocessing as mp
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from av2.utils.io import read_feather, read_city_SE3_ego
from av2.evaluation.tracking.utils import save
from av2.evaluation.tracking.eval import filter_max_dist
from av2.evaluation.scenario_mining.eval import filter_drivable_area
from av2.structures.cuboid import CuboidList
from refAV.paths import AV2_DATA_DIR, SM_DATA_DIR
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
    unique_log_ids = df["log_id"].unique()
    print(f"Found {len(unique_log_ids)} unique log IDs")

    # Columns to exclude
    exclude_columns = ["log_id", "prompt", "mining_category"]

    # Process each log_id
    for log_id in tqdm(unique_log_ids):
        # Create directory for this log_id
        log_dir = base_dir / str(log_id)
        log_dir.mkdir(exist_ok=True, parents=True)

        # Get all entries for this log_id
        log_data = df[df["log_id"] == log_id]

        log_prompts = log_data["prompt"].unique()
        log_data = log_data[log_data["prompt"] == log_prompts[0]]

        # Keep only the "others" columns
        filtered_data = log_data.drop(columns=exclude_columns)

        # Save to a feather file
        output_path = log_dir / "sm_annotations.feather"
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
    # Filtering can dramatically speed up evaluation

    id_stats = {}
    kept_ids = []
    kept_ids_per_timestamp = {}

    for frame in track_data:
        timestamp = frame["timestamp_ns"]
        ids = frame["track_id"]
        scores = frame["score"]
        categories = frame["label"]

        kept_ids_per_timestamp[timestamp] = []

        for index, id in enumerate(ids):

            if id not in id_stats:
                # Track length, min confidence
                id_stats[id] = ([timestamp], scores[index], categories[index])
            else:
                # Weight longer tracks higher
                # print(id_stats[id])
                id_stats[id] = (
                    id_stats[id][0] + [timestamp],
                    scores[index] + id_stats[id][1],
                    categories[index],
                )

    id_stats_by_category = {}
    for id, stats in id_stats.items():
        id_timestamps, score, category = stats

        if category not in id_stats_by_category:
            id_stats_by_category[category] = []
        id_stats_by_category[category].append((id, score))

    for category, category_ids in id_stats_by_category.items():
        sorted_ids = sorted(category_ids, key=lambda row: row[1], reverse=True)

        if category in [
            "REGULAR_VEHCILE",
            "PEDESTRIAN",
            "BOLLARD",
            "CONSTRUCTION_CONE",
            "CONSTRUCTION_BARREL",
        ]:
            topk = 200
        else:
            topk = 100
        for i in range(min(topk, len(sorted_ids))):
            # print(sorted_ids[i][1])
            kept_ids.append(sorted_ids[i][0])
            id_timestamps = id_stats[sorted_ids[i][0]][0]
            for timestamp in id_timestamps:
                kept_ids_per_timestamp[timestamp].append(sorted_ids[i][0])

        # Make sure that all frames have at least one kept id
        for timestamp in kept_ids_per_timestamp.keys():
            while len(kept_ids_per_timestamp[timestamp]) == 0 and i < len(sorted_ids):
                if timestamp in id_stats[sorted_ids[i][0]][0]:
                    kept_ids.append(sorted_ids[i][0])
                    for id_timestamp in id_stats[sorted_ids[i][0]][0]:
                        kept_ids_per_timestamp[id_timestamp].append(sorted_ids[i][0])
                i += 1

    # print(f'filtered from {len(id_stats.keys())} to {len(kept_ids)} ids')

    return kept_ids


def process_sequences(log_id, track_data, dataset_dir, base_output_dir, filter=True):
    """Convert tracker data in AV2 tracking challenge submission format back to AV2 annotation format."""

    split = get_log_split(log_id)
    ego_poses = read_city_SE3_ego(dataset_dir / split / log_id)
    rows = []

    if filter and "score" in track_data[0]:
        kept_ids = filter_ids_by_score(track_data)

    for frame in track_data:
        timestamp = frame["timestamp_ns"]

        # Extract size dimensions
        lengths = frame["size"][:, 0]  # First column for length
        widths = frame["size"][:, 1]  # Second column for width
        heights = frame["size"][:, 2]  # Third column for height

        # Get translations
        city_to_ego = ego_poses[timestamp].inverse()

        city_coords = frame["translation_m"]
        ego_coords = city_to_ego.transform_from(city_coords)

        # Get yaws
        yaws = frame["yaw"]

        # Get categories (names)
        categories = frame["name"]

        # Get track IDs
        track_ids = frame["track_id"]

        if "score" in frame:
            scores = frame["score"]
        else:
            kept_ids = track_ids

        # Process each object in the frame
        ego_yaws = np.zeros(len(track_ids))
        for i in range(len(track_ids)):
            if filter and track_ids[i] not in kept_ids:
                continue

            city_rotation = Rotation.from_euler("xyz", [0, 0, yaws[i]]).as_matrix()

            ego_yaws[i] = Rotation.from_matrix(
                city_to_ego.rotation @ city_rotation
            ).as_euler("zxy")[0]
            # Convert yaw to quaternion
            qw, qx, qy, qz = euler_to_quaternion(ego_yaws[i])

            # Le3DE2E tracker uses
            tz_adjustment = 0
            if "Le3DE2" in str(base_output_dir):
                tz_adjustment = heights[i] / 2

            row = {
                "timestamp_ns": timestamp,
                "track_uuid": str(track_ids[i]),  # Convert track ID to string
                "category": categories[i],
                "length_m": lengths[i],
                "width_m": widths[i],
                "height_m": heights[i],
                "qw": qw,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "tx_m": ego_coords[i, 0],
                "ty_m": ego_coords[i, 1],
                "tz_m": ego_coords[i, 2] + tz_adjustment,
                "num_interior_pts": 1,  # Default value as this info isn't in the original data
            }

            if "score" in frame:
                row["score"] = scores[i]

            rows.append(row)

    if rows:
        # Create DataFrame
        df = pd.DataFrame(rows)

        # Ensure columns are in the correct order
        columns = [
            "timestamp_ns",
            "track_uuid",
            "category",
            "length_m",
            "width_m",
            "height_m",
            "qw",
            "qx",
            "qy",
            "qz",
            "tx_m",
            "ty_m",
            "tz_m",
            "num_interior_pts",
            "score",
        ]
        df = df[columns]

        # Create directory structure and save feather file
        log_dir = base_output_dir / str(log_id)
        log_dir.mkdir(parents=True, exist_ok=True)

        output_path = log_dir / "sm_annotations.feather"
        df.to_feather(output_path)
        # print(f"Created feather file: {output_path}")

        add_ego_to_annotation(output_path.parent, output_path.parent)
        # output_path.unlink()

    # print(f'Tracking predictions processed for log-id {log_id}')


def pickle_to_feather(dataset_dir, input_pickle_path, base_output_dir="output"):
    """
    Convert pickle file to feather files with the specified format.
    Creates a separate feather file for each log_id in its own directory.

    Args:
        input_pickle_path: Path to the input pickle file
        base_output_dir: Base directory for output folders
    """

    # print(f"Reading pickle file: {input_pickle_path}")

    with open(input_pickle_path, "rb") as f:
        data = pickle.load(f)

    with mp.Pool(max(1, int(0.9 * (os.cpu_count())))) as pool:
        pool.starmap(
            process_sequences,
            [
                (log_id, track_data, dataset_dir, base_output_dir)
                for log_id, track_data in data.items()
            ],
        )


def add_ego_to_annotation(log_dir: Path, output_dir: Path = Path("output")):

    split = get_log_split(log_dir)
    annotations_df = read_feather(log_dir / "sm_annotations.feather")
    ego_df = read_feather(
        AV2_DATA_DIR / split / log_dir.name / "city_SE3_egovehicle.feather"
    )
    ego_df["log_id"] = log_dir.name
    ego_df["track_uuid"] = "ego"
    ego_df["category"] = "EGO_VEHICLE"
    ego_df["length_m"] = 4.877
    ego_df["width_m"] = 2
    ego_df["height_m"] = 1.473
    ego_df["qw"] = 1
    ego_df["qx"] = 0
    ego_df["qy"] = 0
    ego_df["qz"] = 0
    ego_df["tx_m"] = 0
    ego_df["ty_m"] = 0
    ego_df["tz_m"] = 0
    ego_df["num_interior_pts"] = 1

    if "score" in annotations_df.columns:
        ego_df["score"] = 1

    synchronized_timestamps = annotations_df["timestamp_ns"].unique()
    ego_df = ego_df[ego_df["timestamp_ns"].isin(synchronized_timestamps)]

    combined_df = pd.concat([annotations_df, ego_df], ignore_index=True)
    feather.write_feather(combined_df, output_dir / "sm_annotations.feather")
    # print(f'Successfully added ego to annotations for log {log_dir.name}.')


def feather_to_csv(feather_path, output_dir):
    df: pd.DataFrame = feather.read_feather(feather_path)
    output_filename = output_dir + "/output.csv"
    df.to_csv("output.csv", index=False)
    # print(f'Successfully saved to {output_filename}')


def mining_category_from_df(df: pd.DataFrame, mining_category: str):

    log_timestamps = np.sort(df["timestamp_ns"].unique())
    category_df = df[df["mining_category"] == mining_category]

    # Keys timestamps, values list of track_uuids
    category_objects = {}

    for timestamp in log_timestamps:
        timestamp_uuids = category_df[category_df["timestamp_ns"] == timestamp][
            "track_uuid"
        ].unique()
        category_objects[timestamp] = list(timestamp_uuids)

    return category_objects


def _convert_log_prompt_df_star(args):
    return convert_log_prompt_df(*args)


def convert_log_prompt_df(
    log_id,
    prompt,
    lpp_df,
    output_dir,
):
    """Process a single log_id and prompt combination."""
    output_path = output_dir / log_id / f"{prompt}.pkl"
    if output_path.exists():
        print(f"Scenario pkl file for {prompt}_{log_id[:8]} already exists.")
        return

    frames = []

    split = get_log_split(Path(log_id))
    log_dir = SM_DATA_DIR / split / log_id
    (output_dir / log_id).mkdir(exist_ok=True)

    annotations = read_feather(log_dir / "sm_annotations.feather")
    log_timestamps = np.sort(annotations["timestamp_ns"].unique())
    all_uuids = list(annotations["track_uuid"].unique())
    ego_poses = get_ego_SE3(log_dir)

    referred_objects = mining_category_from_df(lpp_df, "REFERRED_OBJECT")
    related_objects = mining_category_from_df(lpp_df, "RELATED_OBJECT")

    for timestamp in log_timestamps:
        frame = {}
        timestamp_annotations = annotations[annotations["timestamp_ns"] == timestamp]

        timestamp_uuids = list(timestamp_annotations["track_uuid"].unique())
        ego_to_city = ego_poses[timestamp]

        frame["seq_id"] = (log_id, prompt)
        frame["timestamp_ns"] = timestamp
        frame["ego_translation_m"] = list(ego_to_city.translation)
        frame["description"] = prompt

        n = len(timestamp_uuids)
        frame["translation_m"] = np.zeros((n, 3))
        frame["size"] = np.zeros((n, 3), dtype=np.float32)
        frame["yaw"] = np.zeros(n, dtype=np.float32)
        frame["velocity_m_per_s"] = np.zeros((n, 3))
        frame["label"] = np.zeros(n, dtype=np.int32)
        frame["name"] = np.zeros(n, dtype="<U31")
        frame["track_id"] = np.zeros(n, dtype=np.int32)
        frame["score"] = np.ones(n, dtype=np.float32)

        for i, track_uuid in enumerate(timestamp_uuids):
            track_df = timestamp_annotations[
                timestamp_annotations["track_uuid"] == track_uuid
            ]
            cuboid = CuboidList.from_dataframe(track_df)[0]

            if track_df.empty:
                continue

            ego_coords = track_df[["tx_m", "ty_m", "tz_m"]].to_numpy()
            size = track_df[["length_m", "width_m", "height_m"]].to_numpy()
            translation_m = ego_to_city.transform_from(ego_coords)
            yaw = Rotation.from_matrix(
                ego_to_city.compose(cuboid.dst_SE3_object).rotation
            ).as_euler("zxy")[0]

            if (
                timestamp in referred_objects
                and track_uuid in referred_objects[timestamp]
            ):
                category = "REFERRED_OBJECT"
                label = 0
            elif (
                timestamp in related_objects
                and track_uuid in related_objects[timestamp]
            ):
                category = "RELATED_OBJECT"
                label = 1
            else:
                category = "OTHER_OBJECT"
                label = 2

            frame["translation_m"][i, :] = translation_m
            frame["size"][i, :] = size
            frame["yaw"][i] = yaw
            frame["velocity_m_per_s"][i, :] = np.zeros(3)
            frame["label"][i] = label
            frame["name"][i] = category
            frame["track_id"][i] = all_uuids.index(track_uuid)

        frames.append(frame)

    EVALUATION_SAMPLING_FREQUENCY = 5
    frames = frames[::EVALUATION_SAMPLING_FREQUENCY]

    sequences = {(log_id, prompt): frames}

    save(sequences, output_path)
    print(f"Scenario pkl file for {prompt}_{log_id[:8]} saved successfully.")


def create_gt_mining_pkls_parallel(
    scenario_mining_annotations_path, output_dir: Path, num_processes=None
):
    """
    Generates both a pkl file for evaluation in parallel.

    Args:
        scenario_mining_annotations_path: Path to annotations
        output_dir: Path to output directory
        num_processes: Number of CPU cores to use (None = use all available)
    """

    sm_annotations = read_feather(scenario_mining_annotations_path)
    log_ids = sm_annotations["log_id"].unique()

    # Create a list of (log_id, prompt, filtered_df) tuples to process
    tasks = []
    for log_id in tqdm(log_ids, desc="Separating annotation file"):
        log_df = sm_annotations[sm_annotations["log_id"] == log_id]
        (output_dir / log_id).mkdir(exist_ok=True)
        prompts = log_df["prompt"].unique()
        for prompt in prompts:
            lpp_df = log_df[log_df["prompt"] == prompt]
            tasks.append((log_id, prompt, lpp_df, output_dir))

    # If num_processes is not specified, use all available cores
    if num_processes is None:
        num_processes = mp.cpu_count()

    # The number of processes shouldn't exceed the number of tasks
    num_processes = min(num_processes, len(tasks))

    # Use a Pool of workers to process in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_convert_log_prompt_df_star, tasks),
                total=len(tasks),
                desc="Processing log-prompt pairs",
            )
        )

    # Print summary
    print(f"Completed processing {len(results)} log-prompt combinations")
    return results

def create_gt_pkl_file(
    pkl_dir: Path,
    lpp_path: Path,
    output_path: Path,
    max_range_m: int = 50,
    dataset_dir: str = None,
):
    """
    Combines mining_gt_pkl files created in parallel into a single dictionary
    and adds an is_positive field to all frames.

    is_positive is determined by applying the same filtering used during
    evaluation (max range and drivable area ROI) and checking whether any
    referred objects (label == 0) survive:

        True  - frame contains referred objects that survive filtering
        False - frame does not contain any referred objects (before or after)
        None  - frame had referred objects but they were all filtered out (ambiguous)

    Args:
        pkl_dir: Directory containing individual pkl files
                 (structure: pkl_dir/<log_id>/<prompt>.pkl)
        lpp_path: Path to log_prompt_pairs JSON file
        output_path: Path to save the combined pkl file
        max_range_m: Maximum evaluation range for filtering (default 50)
        dataset_dir: Path to dataset split for ROI filtering (None to skip)
    """
    if output_path.exists():
        return output_path

    with open(lpp_path, "r") as file:
        log_prompt_pairs = json.load(file)

    # Step 1: Combine all individual pkl files
    combined_gt = {}
    missing_count = 0
    for log_id, prompts in tqdm(
        list(log_prompt_pairs.items()), desc="Combining pkl files"
    ):
        for prompt in prompts:
            target_pkl = pkl_dir / log_id / f"{prompt}.pkl"
            if not target_pkl.exists():
                missing_count += 1
                print(f"Warning: Missing pkl file {target_pkl}")
                continue
            with open(target_pkl, "rb") as file:
                track_data = pickle.load(file)
            combined_gt.update(track_data)

    if missing_count > 0:
        print(f"Warning: {missing_count} pkl files were missing")

    print(f"Combined {len(combined_gt)} scenarios")

    # Step 2: Create filtered copy to determine which referred objects survive
    filtered_gt = deepcopy(combined_gt)
    filtered_gt = filter_max_dist(filtered_gt, max_range_m)
    if dataset_dir is not None:
        filtered_gt = filter_drivable_area(filtered_gt, dataset_dir)

    # Step 3: Assign is_positive to each frame
    total_positive = 0
    total_negative = 0
    total_ambiguous = 0
    scenarios_became_negative = []

    for seq_id, frames in combined_gt.items():
        filtered_frames = filtered_gt[seq_id]
        scenario_had_positive = False
        scenario_has_positive = False

        for i, frame in enumerate(frames):
            had_referred = len(frame["label"]) > 0 and 0 in frame["label"]
            has_referred = (
                len(filtered_frames[i]["label"]) > 0
                and 0 in filtered_frames[i]["label"]
            )

            if has_referred:
                frame["is_positive"] = True
                total_positive += 1
                scenario_has_positive = True
            elif had_referred:
                frame["is_positive"] = None
                total_ambiguous += 1
            else:
                frame["is_positive"] = False
                total_negative += 1

            if had_referred:
                scenario_had_positive = True

        if scenario_had_positive and not scenario_has_positive:
            scenarios_became_negative.append(seq_id)

    # Step 4: Print debugging stats
    total_timestamps = total_positive + total_negative + total_ambiguous
    print(f"\n{'='*60}")
    print(f"is_positive statistics:")
    print(f"  Total timestamps:  {total_timestamps}")
    print(
        f"  Positive (True):   {total_positive} "
        f"({100*total_positive/max(total_timestamps,1):.1f}%)"
    )
    print(
        f"  Negative (False):  {total_negative} "
        f"({100*total_negative/max(total_timestamps,1):.1f}%)"
    )
    print(
        f"  Ambiguous (None):  {total_ambiguous} "
        f"({100*total_ambiguous/max(total_timestamps,1):.1f}%)"
    )
    print(
        f"\nScenarios that switched from positive to wholly "
        f"negative/ambiguous: {len(scenarios_became_negative)}"
    )
    for seq_id in scenarios_became_negative:
        log_id, prompt = seq_id
        print(f"  {log_id[:8]}... | {prompt}")
    print(f"{'='*60}\n")

    # Step 5: Save combined gt
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(combined_gt, file)

    print(f"Combined GT pkl saved to {output_path}")
    return output_path

if __name__ == "__main__":

    pass