# -*- coding: utf-8 -*-
"""
Processes Argoverse 2 sensor data to detect objects based on text prompts
using GroundingDINO, segment them using SAM 2, estimate 3D bounding boxes
from LiDAR points within the mask, and save results in AV2 format.

Steps:
1. Load GroundingDINO and SAM 2 models.
2. Load text prompts associated with AV2 log IDs.
3. For each log ID and prompt:
    a. Iterate through relevant camera views and timestamps.
    b. Load the corresponding camera image.
    c. Run GroundingDINO to get 2D bounding boxes for the prompt.
    d. For each detected 2D box:
        i. Run SAM 2 to get a segmentation mask.
        ii. Load the corresponding LiDAR sweep.
        iii. Project LiDAR points onto the image.
        iv. Filter LiDAR points using the SAM mask.
        v. Estimate an axis-aligned 3D bounding box in the ego frame.
    e. Aggregate all 3D bounding boxes for the current log/prompt.
    f. Apply tracking (simple assignment based on distance).
    g. Save results to CSV and Feather files.
4. (Optional) Visualize the generated 3D bounding boxes.
"""

import argparse
import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import torch
from pathlib import Path
from PIL import Image
from rich.progress import track
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

# Argoverse 2 imports
from av2.datasets.sensor.constants import RingCameras, StereoCameras
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.splits import TRAIN, TEST, VAL
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.rendering import vector # For visualization monkey-patch
from av2.rendering.video import tile_cameras # For visualization
from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.io import read_city_SE3_ego, read_ego_SE3_sensor, read_feather
from av2.utils.typing import NDArrayFloat

# GroundingDINO imports
from groundingdino.util.inference import load_model as load_groundingdino_model
from groundingdino.util.inference import load_image as load_groundingdino_image
from groundingdino.util.inference import predict as predict_groundingdino
from groundingdino.util.inference import annotate as annotate_groundingdino
from groundingdino.util.inference import Model


# SAM 2 imports
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Monkey-patch draw_line_frustum thickness for visualization ---
try:
    _old_defaults = vector.draw_line_frustum.__defaults__
    vector.draw_line_frustum.__defaults__ = (8, _old_defaults[1]) # make lines thicker
    print("Monkey-patched av2.rendering.vector.draw_line_frustum line thickness.")
except AttributeError:
    print("Could not monkey-patch av2.rendering.vector.draw_line_frustum.")
# --- End Monkey-patch ---

# Set random seed for reproducibility if needed
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Helper Functions ---

def get_log_split(log_id: str, av2_root: Path) -> str | None:
    """Determines the split (train, val, test) for a given log_id."""
    # Check AV2 predefined splits first
    if log_id in TEST: return 'test'
    if log_id in TRAIN: return 'train'
    if log_id in VAL: return 'val'

    # Fallback: Check directory structure (less reliable)
    for split in ['train', 'val', 'test']:
        if (av2_root / split / log_id).exists():
            return split
    print(f"Warning: Could not determine split for log_id: {log_id}")
    return None

def find_closest_image_timestamp(target_lidar_ts: int, image_folder: Path) -> int | None:
    """Finds the timestamp of the JPG image closest to the target LiDAR timestamp."""
    timestamps = []
    min_diff = float('inf')
    closest_ts = None

    for file_path in image_folder.glob("*.jpg"):
        try:
            ts = int(file_path.stem)
            diff = abs(ts - target_lidar_ts)
            if diff < min_diff:
                min_diff = diff
                closest_ts = ts
        except ValueError:
            continue # Skip files with non-integer names

    # Set a threshold for maximum time difference (e.g., 100ms)
    # AV2 lidar is 10Hz (100ms), cameras are often 30Hz (~33ms)
    max_time_diff_ns = 100 * 1e6 # 100 milliseconds
    if closest_ts is not None and min_diff <= max_time_diff_ns:
         return closest_ts
    else:
        #print(f"Warning: No image found within {max_time_diff_ns / 1e6} ms of LiDAR ts {target_lidar_ts} in {image_folder}")
        return None

def compute_axis_aligned_bounding_box(lidar_points: NDArrayFloat) -> tuple:
    """
    Computes an axis-aligned bounding box for a set of LiDAR points.
    Uses DBSCAN to find the main cluster around the medoid.

    Args:
        lidar_points: (N, 3) array of LiDAR points in the ego frame.

    Returns:
        Tuple containing (length_m, width_m, height_m, qw, qx, qy, qz, tx_m, ty_m, tz_m).
        Returns zeros if no points or clustering fails significantly.
    """
    if lidar_points is None or len(lidar_points) < 5: # Need min_samples for DBSCAN
        print("Warning: Insufficient points for bounding box calculation.")
        return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # --- Clustering (Optional but recommended for robustness) ---
    try:
        # Find medoid (point with minimum sum of distances to others)
        dists = cdist(lidar_points, lidar_points, metric='euclidean')
        total_dists = np.sum(dists, axis=1)
        medoid_idx = np.argmin(total_dists)
        medoid = lidar_points[medoid_idx]

        # Cluster using DBSCAN around the medoid region
        # Eps value might need tuning based on expected object density/size
        clustering = DBSCAN(eps=0.7, min_samples=5).fit(lidar_points)
        labels = clustering.labels_

        # Find the cluster containing the medoid (or the largest cluster if medoid is noise)
        medoid_label = labels[medoid_idx]
        if medoid_label != -1: # Medoid is in a cluster
             filtered_points = lidar_points[labels == medoid_label]
        else:
            # Medoid is noise, find the largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(counts) > 0:
                largest_cluster_label = unique_labels[np.argmax(counts)]
                filtered_points = lidar_points[labels == largest_cluster_label]
            else:
                # No clusters found, use all points (might be noisy)
                print("Warning: DBSCAN found no clusters. Using all points.")
                filtered_points = lidar_points
        
        if len(filtered_points) < 3: # Need at least a few points for AABB
             print("Warning: Cluster has too few points after filtering.")
             filtered_points = lidar_points # Revert to all points if cluster is too small

    except Exception as e:
        print(f"Error during clustering: {e}. Using all points.")
        filtered_points = lidar_points
    # --- End Clustering ---

    if len(filtered_points) == 0:
         print("Warning: No points remaining after filtering for bbox.")
         return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    min_point = np.min(filtered_points, axis=0)
    max_point = np.max(filtered_points, axis=0)

    # Dimensions
    length_m = max_point[0] - min_point[0] # X-axis in ego frame (forward)
    width_m  = max_point[1] - min_point[1] # Y-axis in ego frame (left)
    height_m = max_point[2] - min_point[2] # Z-axis in ego frame (up)

    # Center
    tx_m, ty_m, tz_m = (min_point + max_point) / 2.0

    # Orientation (Axis-aligned means identity quaternion in ego frame)
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    # Basic sanity check on dimensions (e.g., max size < 20m)
    if length_m > 20 or width_m > 20 or height_m > 20:
        print(f"Warning: Unusually large AABB dimensions: L={length_m:.2f}, W={width_m:.2f}, H={height_m:.2f}")
        # Optionally return zeros or clamp dimensions
        # return 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return length_m, width_m, height_m, qw, qx, qy, qz, tx_m, ty_m, tz_m


def assign_track_ids(df: pd.DataFrame, max_dist: float = 5.0) -> pd.DataFrame:
    """
    Assigns track IDs to detections across timestamps using Hungarian algorithm.

    Args:
        df: DataFrame with detections, must include 'timestamp_ns', 'tx_m', 'ty_m', 'tz_m'.
            Assumes DataFrame is sorted by timestamp OR will be sorted internally.
        max_dist: Maximum Euclidean distance (meters) to associate detections.

    Returns:
        DataFrame with an added 'track_uuid' column.
    """
    if df.empty:
        df["track_uuid"] = -1
        return df

    df = df.sort_values("timestamp_ns").reset_index(drop=True)
    df["track_uuid"] = -1 # Initialize column
    next_id = 0
    active_tracks = {}  # track_id -> last_position (x,y,z)

    # Group by timestamp AFTER sorting
    for ts, group in df.groupby("timestamp_ns", sort=False): # Don't re-sort groups
        idxs = group.index.values
        dets = group[["tx_m", "ty_m", "tz_m"]].values

        if not active_tracks:
            # Initialize all as new tracks on the first timestamp
            for det_i, row_idx in enumerate(idxs):
                df.loc[row_idx, "track_uuid"] = next_id
                active_tracks[next_id] = dets[det_i]
                next_id += 1
            continue

        # Build cost matrix: (#active_tracks) x (#current_detections)
        track_ids = list(active_tracks.keys())
        if not track_ids: # Should not happen if active_tracks is not empty, but safety check
             # Treat all current detections as new tracks
            for det_i, row_idx in enumerate(idxs):
                df.loc[row_idx, "track_uuid"] = next_id
                active_tracks[next_id] = dets[det_i]
                next_id += 1
            continue

        track_pos = np.array([active_tracks[t] for t in track_ids]) # Use np.array for broadcasting

        # Calculate cost (Euclidean distance)
        cost = cdist(track_pos, dets) # More efficient than manual broadcasting

        # Solve assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_det_indices = set()
        assigned_track_ids = set()

        # Process assignments
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= max_dist:
                tid = track_ids[r]
                det_idx_in_group = c
                row_idx_in_df = idxs[det_idx_in_group]

                df.loc[row_idx_in_df, "track_uuid"] = tid
                active_tracks[tid] = dets[det_idx_in_group] # Update track position
                assigned_det_indices.add(det_idx_in_group)
                assigned_track_ids.add(tid)

        # Create new tracks for unassigned detections
        for det_i, row_idx in enumerate(idxs):
            if det_i not in assigned_det_indices:
                df.loc[row_idx, "track_uuid"] = next_id
                active_tracks[next_id] = dets[det_i]
                next_id += 1

        # Remove inactive tracks (those not assigned in this step) - Optional
        # inactive_track_ids = set(track_ids) - assigned_track_ids
        # for tid in inactive_track_ids:
        #     del active_tracks[tid] # Or mark as inactive if needed for longer gaps

    # Ensure track IDs are integers
    df["track_uuid"] = df["track_uuid"].astype(int)
    return df

# --- SAM 2 Visualization Helpers (Optional) ---
def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_uint8 = mask.astype(np.uint8) # Convert mask to uint8 for cv2 functions
    mask_image = mask_uint8.reshape(h, w, 1) * color.reshape(1, 1, -1)

    if borders:
        try:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours with antialiasing if possible, adjust thickness
            cv2.drawContours(mask_image, contours, -1, (1.0, 1.0, 1.0, 0.7), thickness=2, lineType=cv2.LINE_AA)
        except Exception as e:
            print(f"Could not draw contours: {e}") # Handle potential errors if cv2 fails

    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def visualize_sam_output(image, masks, scores, bbox, filepath):
    """Visualizes the first SAM mask and the input bbox."""
    if  masks is None or scores is None:
        print("No masks to visualize.")
        return

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    # Show only the first (highest score usually) mask from SAM
    show_mask(masks[0], plt.gca())
    show_box(bbox, plt.gca())
    plt.title(f"SAM Output (Score: {scores[0]:.3f})", fontsize=18)
    plt.axis('off')
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)
    plt.close() # Close the figure to free memory

# --- Core Processing Function ---

def process_detection_with_sam(
    log_id: str,
    cam_name: str,
    lidar_timestamp_ns: int,
    bbox_2d: list[int], # [x1, y1, x2, y2] absolute pixel coords
    sam_predictor: SAM2ImagePredictor,
    av2_root: Path,
    log_data_cache: dict, # Pre-loaded poses, camera models etc.
    description,
    args: argparse.Namespace # Pass args for output paths etc.
) -> pd.Series | None:
    """
    Processes a single 2D detection: runs SAM, projects LiDAR, estimates 3D box.

    Args:
        log_id: Log ID string.
        cam_name: Camera name string.
        lidar_timestamp_ns: Timestamp of the LiDAR sweep to use.
        bbox_2d: Bounding box [x1, y1, x2, y2] in absolute pixel coordinates.
        sam_predictor: Initialized SAM2ImagePredictor instance.
        av2_root: Path to the root AV2 dataset directory.
        log_data_cache: Dictionary containing pre-loaded 'pinhole_cams', 'ego_poses', 'split'.
        args: Command line arguments for configuration (e.g., output dirs).

    Returns:
        A Pandas Series with 3D bounding box parameters (length_m, width_m, ...)
        or None if processing fails.
    """
    try:
        split = log_data_cache['split']
        pinhole_cam = log_data_cache['pinhole_cams'].get(cam_name)
        ego_poses = log_data_cache['ego_poses']

        if not pinhole_cam:
            print(f"Error: Pinhole camera model not found for {cam_name} in log {log_id}")
            return None
        if lidar_timestamp_ns not in ego_poses:
             print(f"Error: Ego pose not found for lidar timestamp {lidar_timestamp_ns} in log {log_id}")
             return None

        # 1. Find the corresponding image
        cam_folder = av2_root / split / log_id / "sensors" / "cameras" / cam_name
        image_timestamp_ns = find_closest_image_timestamp(lidar_timestamp_ns, cam_folder)

        if image_timestamp_ns is None:
            #print(f"Skipping detection: No suitable image found for cam {cam_name} near lidar ts {lidar_timestamp_ns}")
            return None
        if image_timestamp_ns not in ego_poses:
             print(f"Error: Ego pose not found for image timestamp {image_timestamp_ns} in log {log_id}")
             return None

        image_path = cam_folder / f"{image_timestamp_ns}.jpg"
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return None

        # 2. Load Image
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            H, W = image_np.shape[:2]
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

        # 3. Run SAM 2
        x1, y1, x2, y2 = bbox_2d
        # Clip box coordinates to image dimensions
        input_box = np.array([
            max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)
        ])

        # Check if box is valid after clipping
        if input_box[0] >= input_box[2] or input_box[1] >= input_box[3]:
             print(f"Warning: Invalid bounding box after clipping: {input_box} for image {W}x{H}. Original: {bbox_2d}")
             return None

        try:
            sam_predictor.set_image(image_np)
            # Use box input, no points, single mask output is often sufficient
            masks, scores, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :], # SAM expects batch dimension
                multimask_output=False, # Get single best mask
            )
        except Exception as e:
            print(f"Error during SAM prediction: {e}")
            return None

        if masks is None or scores is None or masks[0].shape != (H, W):
            print(f"Warning: SAM prediction failed or returned unexpected mask shape for {image_path}")
            return None

        # Optional: Visualize SAM output
        if args.visualize_sam:
            sam_vis_path = Path(args.output_dir) / "sam_visualizations" / log_id / cam_name / description
            sam_vis_file = sam_vis_path / f"{lidar_timestamp_ns}_{image_timestamp_ns}.png"
            visualize_sam_output(image_np, masks, scores, input_box, str(sam_vis_file))


        # 4. Load LiDAR Sweep
        lidar_path = av2_root / split / log_id / "sensors" / "lidar" / f"{lidar_timestamp_ns}.feather"
        if not lidar_path.exists():
            print(f"Error: LiDAR file not found: {lidar_path}")
            return None
        try:
            lidar_df = read_feather(lidar_path)
            # Ensure 'x', 'y', 'z' columns exist
            if not all(col in lidar_df.columns for col in ['x', 'y', 'z']):
                print(f"Error: Missing required columns in LiDAR file: {lidar_path}")
                return None
            lidar_points_ego = lidar_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        except Exception as e:
            print(f"Error reading LiDAR file {lidar_path}: {e}")
            return None


        # 5. Project LiDAR to Image & Filter with Mask
        try:
            city_SE3_ego_cam_t = ego_poses[image_timestamp_ns]
            city_SE3_ego_lidar_t = ego_poses[lidar_timestamp_ns]

            lidar_points_in_image_uv, lidar_points_cam_frame, is_valid_projection = \
                pinhole_cam.project_ego_to_img_motion_compensated(
                    points_lidar_time=lidar_points_ego,
                    city_SE3_ego_cam_t=city_SE3_ego_cam_t,
                    city_SE3_ego_lidar_t=city_SE3_ego_lidar_t
                )
        except KeyError as e:
             print(f"Error: Missing pose for timestamp {e} during projection.")
             return None
        except Exception as e:
            print(f"Error during LiDAR projection: {e}")
            return None

        # Get the boolean mask from SAM (should be HxW)
        sam_mask = masks[0] # Assuming single mask output

        # Round projected coordinates and clip to image bounds
        lidar_u = np.round(lidar_points_in_image_uv[:, 0]).astype(int)
        lidar_v = np.round(lidar_points_in_image_uv[:, 1]).astype(int)

        # Create a boolean mask for points within image boundaries
        u_valid = (lidar_u >= 0) & (lidar_u < W)
        v_valid = (lidar_v >= 0) & (lidar_v < H)
        uv_valid = u_valid & v_valid

        # Combine AV2's valid projection mask and our boundary check
        overall_valid = is_valid_projection & uv_valid

        # Initialize mask for LiDAR points
        bool_lidar_in_sam_mask = np.zeros(len(lidar_points_ego), dtype=bool)

        # Check SAM mask only for valid points within image bounds
        valid_indices = np.where(overall_valid)[0]
        if len(valid_indices) > 0:
            valid_u = lidar_u[valid_indices]
            valid_v = lidar_v[valid_indices]
            # Index into the SAM mask using valid coordinates
            bool_lidar_in_sam_mask[valid_indices] = sam_mask[valid_v, valid_u] == 1 # Assuming SAM mask is boolean or 0/1

        # Get the 3D points in ego frame that fall inside the mask
        lidar_points_in_mask_ego = lidar_points_ego[bool_lidar_in_sam_mask]

        if len(lidar_points_in_mask_ego) == 0:
             #print(f"Warning: No LiDAR points found within the SAM mask for {image_path}")
             return None # Or handle as needed

        # 6. Compute Axis-Aligned 3D Bounding Box
        bbox_3d_params = compute_axis_aligned_bounding_box(lidar_points_in_mask_ego)

        # Check if bbox computation returned valid results (non-zero dimensions)
        if bbox_3d_params[0] <= 0 or bbox_3d_params[1] <= 0 or bbox_3d_params[2] <= 0:
             print(f"Warning: Invalid 3D bbox dimensions computed for {image_path}. Points: {len(lidar_points_in_mask_ego)}")
             return None


        # 7. Return results as a Pandas Series
        return pd.Series(
            bbox_3d_params,
            index=[
                "length_m", "width_m", "height_m",
                "qw", "qx", "qy", "qz",
                "tx_m", "ty_m", "tz_m"
            ]
        )

    except Exception as e:
        print(f"!! Unhandled exception in process_detection_with_sam for log {log_id}, cam {cam_name}, ts {lidar_timestamp_ns}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Visualization Function for 3D Boxes ---

def generate_sensor_dataset_visualizations(
    dataset_dir: Path,
    cuboid_csv: Path,
    output_vis_dir: Path,
    log_id: str,
    description: str,
    frames_to_render: int = 50 # Limit number of frames per log/prompt
    ) -> None:
    """
    Generates videos visualizing the 3D bounding boxes from a CSV file.
    Adapted from AV2 API examples.
    """
    print(f"Generating visualization for {log_id} - {description}...")

    if not cuboid_csv.exists():
        print(f"Error: Cuboid CSV not found: {cuboid_csv}")
        return

    # --- Data Loading ---
    # Determine camera names (using RingCameras as default)
    valid_ring = {x.value for x in RingCameras}
    cam_names_str = tuple(x.value for x in RingCameras)
    cam_enums = [RingCameras(cam) for cam in cam_names_str if cam in valid_ring]
    cam_names_enum = tuple(cam_enums)

    # Load AV2 dataloader for the specific log
    try:
        dataloader = SensorDataloader(
            dataset_dir, # Should be the root AV2 'Sensor' directory
            log_ids=[log_id], # Load only the relevant log
            cam_names=cam_names_enum,
            with_annotations=False, # We load annotations from our CSV
            verbose=False
        )
    except Exception as e:
        print(f"Error initializing SensorDataloader for log {log_id}: {e}")
        return

    # Load cuboids from the generated CSV
    try:
        df = pd.read_csv(cuboid_csv)
        if df.empty:
            print(f"No cuboids found in {cuboid_csv}. Skipping visualization.")
            return
        all_cuboids = []
        for _, row in df.iterrows():
            rotation = R.from_quat([row.qx, row.qy, row.qz, row.qw]).as_matrix()
            ego_SE3_object = SE3(rotation=rotation, translation=np.array([row.tx_m, row.ty_m, row.tz_m]))
            cuboid = Cuboid(
                dst_SE3_object=ego_SE3_object,
                length_m=row.length_m,
                width_m=row.width_m,
                height_m=row.height_m,
                timestamp_ns=int(row.timestamp_ns),
                category=str(row.get('track_uuid', '')) # Use track ID as category for color
            )
            all_cuboids.append(cuboid)
        log_cuboid_list = CuboidList(all_cuboids)
    except Exception as e:
        print(f"Error loading or parsing cuboid CSV {cuboid_csv}: {e}")
        return

    # --- Rendering Loop ---
    output_log_dir = output_vis_dir / f"{log_id}_{description.replace(' ', '_')[:30]}" # Shorten desc for filename
    output_log_dir.mkdir(parents=True, exist_ok=True)
    rendered_count = 0

    for i in track(range(len(dataloader)), description=f"Rendering {log_id[:8]}..."):
        if rendered_count >= frames_to_render:
            print(f"Reached render limit ({frames_to_render}) for {log_id}")
            break

        try:
            datum = dataloader[i]
            lidar_timestamp_ns = datum.timestamp_ns

            # Filter cuboids for the current timestamp
            ts_cuboids = [c for c in log_cuboid_list.cuboids if c.timestamp_ns == lidar_timestamp_ns]
            if not ts_cuboids:
                continue # Skip frames with no detections

            current_cuboid_list = CuboidList(ts_cuboids)
            timestamp_city_SE3_ego_dict = datum.timestamp_city_SE3_ego_dict
            synchronized_imagery = datum.synchronized_imagery

            if synchronized_imagery is None:
                continue

            cam_name_to_img = {}
            valid_frame = True
            for cam_name_enum in cam_names_enum:
                cam_name_str = cam_name_enum.value
                if cam_name_str in synchronized_imagery:
                    cam_data = synchronized_imagery[cam_name_str]
                    if cam_data.timestamp_ns in timestamp_city_SE3_ego_dict:
                        city_SE3_ego_cam_t = timestamp_city_SE3_ego_dict[cam_data.timestamp_ns]
                        img = cam_data.img.copy()

                        # Project cuboids onto this camera image
                        img = current_cuboid_list.render_on_image(
                            img,
                            cam_data.camera_model,
                            city_SE3_ego_t=city_SE3_ego_cam_t,
                            city_SE3_ego_rendering_t=city_SE3_ego_cam_t, # Assuming rendering at camera time
                            thickness=2 # Use the monkey-patched default or set explicitly
                        )
                        cam_name_to_img[cam_name_str] = img
                    else:
                        #print(f"Missing pose for camera {cam_name_str} at ts {cam_data.timestamp_ns}")
                        valid_frame = False
                        break
                else:
                    #print(f"Missing synchronized image for camera {cam_name_str}")
                    valid_frame = False
                    break
            
            if not valid_frame or len(cam_name_to_img) != len(cam_names_enum):
                continue # Skip if any camera is missing or poses are missing

            # Tile the images
            tiled_img = tile_cameras(cam_name_to_img, list(cam_names_str)) # Pass camera order explicitly

            # Save the frame
            out_path = output_log_dir / f"{lidar_timestamp_ns}.png"
            cv2.imwrite(str(out_path), tiled_img)
            rendered_count += 1

        except Exception as e:
            print(f"\nError during rendering frame {i} for log {log_id}: {e}")
            import traceback
            traceback.print_exc()
            continue # Try next frame

    print(f"Finished visualization for {log_id} - {description}. Frames saved in {output_log_dir}")


# --- Main Execution ---

def main(log_id, gpu_id, args):
    """Main function to run the detection and processing pipeline."""

    # --- Setup ---
    av2_root = Path(args.av2_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Models ---
    print("Loading GroundingDINO model...")
    groundingdino_model = load_groundingdino_model(
        args.groundingdino_config, args.groundingdino_weights, device=device
    )
    print("Loading SAM 2 model...")
    sam_predictor = SAM2ImagePredictor.from_pretrained(args.sam_model_type, device=device)
    print("Models loaded.")

    # --- Load Prompts ---
    lpp_path = Path(args.log_prompt_pairs)
    if not lpp_path.exists():
        raise FileNotFoundError(f"Log prompt pairs file not found: {lpp_path}")
    with open(lpp_path, 'r') as f:
        log_prompt_pairs = json.load(f)

    # --- Get list of all logs in the dataset ---
    all_log_ids = set()
    for split in ['test', 'val']:
        split_dir = av2_root / split
        if split_dir.is_dir():
             all_log_ids.update(p.name for p in split_dir.iterdir() if p.is_dir())
    print(f"Found {len(all_log_ids)} logs in {av2_root}")


    with open('baselines/groundingSAM/log_id_to_start_index.json', 'rb') as file:
        log_id_to_start_index = json.load(file)

    # --- Processing Loop ---
    processed_logs = 0
    dataloader = SensorDataloader(av2_root, with_annotations=False, with_cache=True)

    prompts = log_prompt_pairs[log_id]

    if log_id not in all_log_ids:
            print(f"Skipping log {log_id}: Not found in dataset structure at {av2_root}")
            return
    
    print(f"\n--- Processing Log: {log_id} ---")
    processed_logs += 1

    # Pre-load log-specific data to avoid repeated reading
    split = get_log_split(log_id, av2_root)
    if not split:
        print(f"Could not determine split for {log_id}, skipping.")
        return
    log_path = av2_root / split / log_id
    try:
        ego_poses = read_city_SE3_ego(log_path)
        sensor_extrinsics = read_ego_SE3_sensor(log_path)
        pinhole_cams = {}
        # Use RingCameras by default, adapt if using others
        cam_names_to_load = [c.value for c in RingCameras]
        for cam_name in cam_names_to_load:
                if cam_name in sensor_extrinsics: # Check if camera exists for this log
                    try:
                        pinhole_cams[cam_name] = PinholeCamera.from_feather(log_path, cam_name)
                    except FileNotFoundError:
                        print(f"Warning: Calibration file not found for camera {cam_name} in log {log_id}. Skipping this camera.")
                    except Exception as e:
                        print(f"Warning: Error loading camera model for {cam_name} in log {log_id}: {e}. Skipping this camera.")

        if not pinhole_cams:
                print(f"Error: No valid camera models loaded for log {log_id}. Skipping.")
                return

        log_data_cache = {
            'split': split,
            'ego_poses': ego_poses,
            'pinhole_cams': pinhole_cams,
            'cam_names': list(pinhole_cams.keys()) # Store names of successfully loaded cams
        }
    except FileNotFoundError as e:
        print(f"Error loading essential data for log {log_id} (e.g., poses): {e}. Skipping.")
        return
    except Exception as e:
            print(f"Unexpected error loading data for log {log_id}: {e}. Skipping.")
            return

    for prompt_idx, description in enumerate(prompts):
        print(f"  Prompt {prompt_idx + 1}/{len(prompts)}: '{description}'")

        # Define output file paths for this specific log/prompt
        desc_safe = re.sub(r'[^\w\-]+', '_', description).strip('_').lower()[:50] # Make description filename-safe
        output_stem = f"detections_{log_id}_{desc_safe}"
        output_csv = output_dir / f"{output_stem}.csv"
        output_feather = output_dir / f"{output_stem}.feather"
        output_vis_dir = output_dir / "visualizations_3d"

        if output_csv.exists() and output_feather.exists() and not args.overwrite:
            print(f"    Output files exist, skipping generation.")
            # Still run visualization if requested and vis output doesn't exist
            vis_exists = (output_vis_dir / f"{log_id}_{desc_safe}").exists()
            if args.visualize_output and not vis_exists:
                    generate_sensor_dataset_visualizations(
                        av2_root, output_csv, output_vis_dir, log_id, desc_safe, args.vis_frames
                    )
            continue # Skip to next prompt

        all_detections_3d = []
        processed_frames = 0

        i = log_id_to_start_index[log_id]
        datum = dataloader[i]
        
        # Iterate through LiDAR timestamps as the primary time reference
        while datum.log_id == log_id:
            print(f'{i-log_id_to_start_index[log_id]}/155', end='\r')

            # Process each camera view for this timestamp
            for cam_name in log_data_cache['cam_names']:
                try:
                    image = datum.synchronized_imagery[cam_name].img
                except:
                    print(f'Image not found for {cam_name}!')
                    continue

                timestamp_ns = datum.timestamp_ns

                try:

                    image_tensor = Model.preprocess_image(image)
                    # GroundingDINO needs image_source (path or array), image (tensor)
                    image_tensor = image_tensor.to(device)

                    boxes_filt, logits_filt, phrases_filt = predict_groundingdino(
                        model=groundingdino_model,
                        image=image_tensor,
                        caption=description, # Use the current prompt
                        box_threshold=args.box_threshold,
                        text_threshold=args.text_threshold,
                        device=device
                    )

                    # Optional: Visualize GroundingDINO output
                    if args.visualize_groundingdino and len(boxes_filt) > 0:
                        annotated_frame = annotate_groundingdino(
                            image_source= image, boxes=boxes_filt,
                            logits=logits_filt, phrases=phrases_filt
                        )
                        gd_vis_path = Path(output_dir) / "groundingdino_visualizations" / log_id / cam_name
                        gd_vis_path.mkdir(parents=True, exist_ok=True)
                        gd_vis_file = gd_vis_path / f"{timestamp_ns}.png"
                        cv2.imwrite(str(gd_vis_file), annotated_frame)

                except Exception as e:
                    print(f"\nError during GroundingDINO prediction for {cam_name} at iteration {i}: {e}")
                    continue # Skip to next camera/frame on error


                # Process each detection with SAM
                H, W = image.shape[:2] # Get image dimensions from loaded source
                for box_norm in boxes_filt.cpu().numpy(): # Iterate through detected boxes
                    # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
                    cx, cy, w, h = box_norm
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    bbox_2d_abs = [x1, y1, x2, y2]

                    # Call the SAM processing function
                    result_3d = process_detection_with_sam(
                        log_id, cam_name, timestamp_ns, bbox_2d_abs,
                        sam_predictor, av2_root, log_data_cache, description, args
                    )

                    # If successful, add to our list of detections
                    if result_3d is not None:
                        detection_data = {
                            'log_id': log_id,
                            'timestamp_ns': timestamp_ns,
                            # Add prompt info if needed later
                            #'prompt': description,
                            #'camera': cam_name,
                        }
                        detection_data.update(result_3d.to_dict())
                        all_detections_3d.append(detection_data)

            processed_frames +=1
            i += 5
            datum = dataloader[i]


        # --- Post-Processing for the current log/prompt ---
        if not all_detections_3d:
            print(f"    No valid 3D detections generated for this prompt.")
            # Create empty files to mark as processed
            pd.DataFrame(columns=[ # Ensure schema matches expected output
                'log_id', 'timestamp_ns', 'track_uuid',
                'length_m', 'width_m', 'height_m',
                'qw', 'qx', 'qy', 'qz',
                'tx_m', 'ty_m', 'tz_m']).to_csv(output_csv, index=False)
            pd.DataFrame(columns=[
                    'log_id', 'timestamp_ns', 'track_uuid',
                'length_m', 'width_m', 'height_m',
                'qw', 'qx', 'qy', 'qz',
                'tx_m', 'ty_m', 'tz_m']).to_feather(output_feather)
            continue # Skip to next prompt

        # Convert list of dicts to DataFrame
        df_output = pd.DataFrame(all_detections_3d)

        # Assign Track IDs
        print(f"    Assigning track IDs...")
        df_output = assign_track_ids(df_output, max_dist=args.tracking_max_dist)

        # Select and order columns for final output
        output_columns = [
            'log_id', 'timestamp_ns', 'track_uuid',
            'length_m', 'width_m', 'height_m',
            'qw', 'qx', 'qy', 'qz',
            'tx_m', 'ty_m', 'tz_m'
        ]
        # Ensure all columns exist, fill missing with defaults if necessary
        for col in output_columns:
            if col not in df_output.columns:
                if col == 'track_uuid': df_output[col] = -1
                else: df_output[col] = 0.0 # Or appropriate default
        
        df_final = df_output[output_columns]


        # Save to CSV and Feather
        print(f"    Saving {len(df_final)} detections to {output_csv} and {output_feather}")
        try:
                df_final.to_csv(output_csv, index=False)
                df_final.to_feather(output_feather)
        except Exception as e:
            print(f"Error saving output files: {e}")

        # Optional: Generate Visualization Video/Frames
        if args.visualize_output:
                generate_sensor_dataset_visualizations(
                    av2_root, output_csv, output_vis_dir, log_id, desc_safe, args.vis_frames
                )

    del groundingdino_model, sam_predictor, dataloader
    torch.cuda.empty_cache()

    print(f"\n--- Finished processing {processed_logs} logs ---")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GroundingDINO + SAM 2 on Argoverse 2 data.")

    import multiprocessing as mp
    mp.set_start_method('spawn')

    # --- Paths ---
    parser.add_argument("--av2_root", type=str, default="/data3/shared/datasets/ArgoVerse2/Sensor", help="Path to the root Argoverse 2 Sensor dataset directory.")
    parser.add_argument("--output_dir", type=str, default="baselines/groundingSAM/output", help="Directory to save output files (CSV, Feather, visualizations).")
    parser.add_argument("--log_prompt_pairs", type=str, default="av2_sm_downloads/log_prompt_pairs_test.json", help="Path to the JSON file mapping log_ids to lists of text prompts.")
    parser.add_argument("--split", type=str, default="test", help="Scenario mining evaluation split")
    parser.add_argument("--groundingdino_config", type=str, default="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Path to GroundingDINO model config file.")
    parser.add_argument("--groundingdino_weights", type=str, default="/home/crdavids/Trinity-Sync/GroundingDINO/weights/groundingdino_swint_ogc.pth", help="Path to GroundingDINO model weights file.")
    # Find available SAM types from https://github.com/facebookresearch/segment-anything-2
    parser.add_argument("--sam_model_type", type=str, default="facebook/sam2-hiera-large", help="HuggingFace model identifier for SAM 2 predictor (e.g., 'facebook/sam2-hiera-large').")

    # --- Detection/Processing Parameters ---
    parser.add_argument("--box_threshold", type=float, default=0.35, help="GroundingDINO box confidence threshold.")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text confidence threshold.")
    parser.add_argument("--tracking_max_dist", type=float, default=5.0, help="Maximum distance (meters) for associating detections in tracking.")
    parser.add_argument("--max_frames_per_prompt", type=int, default=-1, help="Maximum number of LiDAR timestamps (frames) to process per prompt per log (-1 for all).")


    # --- Control Flow & Visualization ---
    parser.add_argument("--overwrite", default=False, help="Overwrite existing output CSV/Feather files.")
    parser.add_argument("--visualize_groundingdino", default=False, help="Save annotated images from GroundingDINO.")
    parser.add_argument("--visualize_sam", default=False, help="Save annotated images with SAM masks.")
    parser.add_argument("--visualize_output", default=False, help="Generate 3D bounding box visualizations on camera images after processing.")
    parser.add_argument("--vis_frames", type=int, default=50, help="Maximum number of frames to render for 3D visualization per log/prompt.")

    args = parser.parse_args()

    # --- Basic Validation ---
    if not Path(args.log_prompt_pairs).is_file():
        parser.error(f"Log prompt pairs file not found: {args.log_prompt_pairs}")
    if not Path(args.groundingdino_config).is_file():
         parser.error(f"GroundingDINO config not found: {args.groundingdino_config}")
    if not Path(args.groundingdino_weights).is_file():
         parser.error(f"GroundingDINO weights not found: {args.groundingdino_weights}")
    # SAM model type is checked during loading by HuggingFace library

    lpp_path = Path(args.log_prompt_pairs)
    with open(lpp_path, 'r') as f:
        log_prompt_pairs = json.load(f)

    num_gpus = torch.cuda.device_count()
    log_ids = list(log_prompt_pairs.keys())
    np.random.shuffle(log_ids)

    with mp.Pool(processes=os.cpu_count()//8) as pool:
        pool.starmap(main, [(log_id, i%num_gpus, args) for i, log_id in enumerate(log_ids)], chunksize=1)
