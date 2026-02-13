"""Script to cache the color of nuScenes tracked objects."""
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from copy import deepcopy
import multiprocessing as mp
from refAV.utils import read_feather, get_best_crop, get_img_crop, get_clip_colors, EasyDataLoader
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import json
from PIL import Image
import traceback
from transformers import pipeline

nuscenes_tracking_path = Path(
    "output/tracker_predictions/PFTrack_FullRes_Tracking/nuprompt_val"
)
crop_dir = Path("output/visualization/pftrack2_crops")

# dict[log_id][track_id] = str(color)
infos_batches = []
image_batches = []
batch_size = 256

count_in_batch = 0
current_batch = []
current_infos = []

try:
    with open("output/cache/color_cache.json", "rb") as file:
        colors = json.load(file)
except:
    colors = {}

dataloader = EasyDataLoader(Path('output/tracker_predictions/PFTrack_Tracking'))

#for log_dir in nuscenes_tracking_path.iterdir():
#    if (log_dir/'cache/track_crop_information.json').exists():
#        (log_dir/'cache/track_crop_information.json').unlink()

def save_image_crops(log_dir, dataloader):

    if not log_dir.is_dir():
        return

    df = pd.read_feather(log_dir/'sm_annotations.feather')
    track_uuids = df['track_uuid'].unique()

    for track_uuid in track_uuids:
        crop_path = crop_dir/log_dir.name/f'{track_uuid}.png'
        if crop_path.exists():
            continue

        best_crop = get_best_crop(track_uuid, log_dir)

        if best_crop is None:
            continue

        img_path = dataloader.get_closest_img_fpath(log_dir.name, best_crop['cam'], best_crop['timestamp'])
        img = Image.open(img_path)
        crop = img.crop(best_crop['crop'])


        crop_path.parent.mkdir(exist_ok=True, parents=True)
        crop.save(crop_path)

with mp.Pool(int(.5*mp.cpu_count())) as pool:
    pool.starmap(save_image_crops, [(log_dir, dataloader) for log_dir in list(nuscenes_tracking_path.iterdir())])





num_unlabeled = 0
for log_dir in tqdm(list(nuscenes_tracking_path.iterdir())):
    if not log_dir.is_dir():
        continue

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
    for crop_path in log_dir.iterdir():

        log_id = log_dir.stem
        track_uuid = crop_path.stem

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

for log_dir in nuscenes_tracking_path.iterdir():
    if not log_dir.is_dir():
        continue
    
    cache_path = log_dir/'cache/color_cache.json'
    if not cache_path.exists():
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        
    with open(log_dir/'cache/color_cache.json', 'w') as file:
        json.dump(colors[log_dir.name], file, indent=4)