"""Script to cache the SigLIP precited color of StreamPETR tracked objects. Use to speed up RefAV evaluation."""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from copy import deepcopy
from refAV.utils import read_feather, get_best_crop, get_img_crop, get_clip_colors
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import json
import traceback
from transformers import pipeline

stream_petr_path = Path(
    "output/tracker_predictions/StreamPETR_Tracking/nuprompt_val_large"
)
crop_dir = Path("output/visualization/nuprompt_score_crops")

# dict[log_id][track_id] = str(color)
infos_batches = []
image_batches = []
batch_size = 512

count_in_batch = 0
current_batch = []
current_infos = []


try:
    with open("output/cache/color_cache.json", "rb") as file:
        colors = json.load(file)
except:
    colors = {}


num_unlabeled = 0
for log_dir in tqdm(list(stream_petr_path.iterdir())):
    if log_dir.stem not in colors:
        print(f"Missing log {log_dir.stem}")
        colors[log_dir.stem] = {}

    df = read_feather(log_dir / "sm_annotations.feather")
    for track_uuid in df["track_uuid"]:
        if str(track_uuid) not in colors[log_dir.stem]:
            colors[log_dir.stem][track_uuid] = None
            num_unlabeled += 1

print(num_unlabeled)
with open("output/cache/color_cache.json", "w") as file:
    json.dump(colors, file, indent=4)


for log_dir in tqdm(list(crop_dir.iterdir())):
    for track_dir in log_dir.iterdir():
        for crop_path in track_dir.iterdir():

            log_id = log_dir.stem
            track_uuid = track_dir.stem

            if (
                log_id in colors
                and track_uuid in colors[log_id]
                and colors[log_id][track_uuid] is not None
            ):
                continue

            if count_in_batch < batch_size:
                current_infos.append((log_id, track_uuid))  # log_id, track_uuid
                current_batch.append(str(crop_path.resolve()))
                count_in_batch += 1
            else:
                image_batches.append(deepcopy(current_batch))
                infos_batches.append(deepcopy(current_infos))
                current_batch = []
                current_infos = []
                count_in_batch = 0

ckpt = "google/siglip2-so400m-patch16-naflex"
pipe = pipeline(
    model=ckpt,
    task="zero-shot-image-classification",
)

possible_colors = ["white", "silver", "black", "red", "yellow", "blue"]
for image_batch, batch_info in tqdm(
    zip(image_batches, infos_batches), total=len(image_batches)
):
    batch_colors = get_clip_colors(image_batch, possible_colors, pipe=pipe)

    for color, (log_id, track_uuid) in zip(batch_colors, batch_info):
        if log_id not in colors:
            colors[log_id] = {}
        if track_uuid not in colors[log_id]:
            colors[log_id][track_uuid] = {}
        colors[log_id][track_uuid] = color

    with open("output/cache/color_cache.json", "w") as file:
        json.dump(colors, file, indent=4)