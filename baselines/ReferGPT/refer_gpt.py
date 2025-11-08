import openai
import pandas as pd
import numpy as np
from transformers import (
    AutoModel,
    AutoProcessor,
    Siglip2Model,
    Siglip2Processor,
    Qwen3VLProcessor,
)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from pathlib import Path
from copy import deepcopy
import json
import multiprocessing as mp

from refAV.utils import (
    get_all_crops,
    get_nth_pos_deriv,
    get_nth_yaw_deriv,
    get_ego_uuid,
    get_uuids_of_category,
    get_eval_timestamps,
    create_mining_pkl,
    get_img_crop,
    get_best_crop,
)
from refAV.atomic_functions import output_scenario


def clip_similarity(query: str, captions: list[str | Image.Image], batch_size=32):

    model_id = "google/siglip2-so400m-patch16-naflex"
    siglip: Siglip2Model = AutoModel.from_pretrained(model_id)
    processor: Siglip2Processor = AutoProcessor.from_pretrained(model_id)

    with torch.no_grad():

        iter = 0
        texts = [query] + captions
        text_features = []
        while tqdm(iter < len(texts)):
            batch_texts = texts[iter : min(len(texts), iter + batch_size)]
            inputs = processor(text=batch_texts).to(siglip.device)
            text_features.append(
                siglip.get_text_features(**inputs)
            )  # I assume output of text_features is [B, H] tensor

        text_features = np.concat(
            text_features, axis=0
        )  # Convert [NB, B, H] list of tensors to [N+1, H] tensor
        query_features = text_features[0].unsqueeze(0) / torch.linalg.norm(
            text_features[0]
        )
        caption_features = text_features[1:] / torch.linalg.norm(
            text_features[1:], axis=1
        )

        clip_similarity = query_features @ caption_features.T

    return clip_similarity


def get_object_captions(
    cropped_images_by_object_timestamp,
    motion_by_object_timestamp,
    log_dir,
    llm_prompt_path='baselines/ReferGPT/prompt.txt',
    device="cuda:0",
    batch_size=32
)->dict:

    with open(llm_prompt_path, "r") as file:
        prompt_template = file.read()

    captions_by_object_timestamp = {}

    cur_batch = []
    message_batches = []
    caption_info_list = []
    for track_uuid, cropped_images_by_timestamp in cropped_images_by_object_timestamp.items():
        if track_uuid not in captions_by_object_timestamp:
            captions_by_object_timestamp[track_uuid] = {}

        for timestamp, cropped_image in cropped_images_by_object_timestamp.items():
            caption_path = log_dir/f'cache/captions/{track_uuid}/{int(timestamp)}.txt'
            if caption_path.exists():
                captions_by_object_timestamp[track_uuid][timestamp] = (
                    open(caption_path).read().format()
                )
            else:
                chat = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": cropped_image,
                            },
                            {
                                "type": "text",
                                "text": prompt_template.format(
                                    object_motion=motion_by_object_timestamp[
                                        track_uuid
                                    ][timestamp]
                                ),
                            },
                        ],
                    }
                ]

                if len(cur_batch) < batch_size:
                    cur_batch.append(chat)
                else:
                    message_batches.append(deepcopy(cur_batch))
                    cur_batch = []

            caption_info_list.append((caption_path.exists(), track_uuid, timestamp))

    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device=device,
        attn_implementation="flash_attention_2",
    )
    processor: Qwen3VLProcessor = AutoProcessor.from_pretrained(model_id)

    caption_index = 0
    for batch in tqdm(message_batches):
        inputs = processor.apply_chat_template(
            batch,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        decoded_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        
        for caption in decoded_text:
            while caption_info_list[caption_index][0]:
                caption_index += 1

            _, track_uuid, timestamp = caption_info_list[caption_index]
            captions_by_object_timestamp[track_uuid][timestamp] = caption
            caption_path = log_dir/f'cache/captions/{track_uuid}/{int(timestamp)}.txt'
            open(caption_path, "w").write(caption)
            caption_index += 1

    return captions_by_object_timestamp


def fuzzy_matching(query: str, captions: list[str]) -> list[float]:
    """Calculate token match score with fuzzy matching."""

    fuzzy_matching_scores = []
    for prompt in captions:

        query = query.replace("-", " ")
        stop_words = set(stopwords.words("english"))
        stop_words -= {"same"}
        stop_words |= {"color", "the", "which", "are", "of", "who", "a"}

        def tokenize_and_filter(sentence):
            tokens = word_tokenize(sentence.lower())
            return [token for token in tokens if token not in stop_words]

        tokens1 = tokenize_and_filter(query)
        tokens2 = tokenize_and_filter(prompt)

        def fuzzy_token_match(token1, token2):
            return SequenceMatcher(None, token1, token2).ratio()

        matched_tokens = set()
        overlap_score = 0
        for token1 in tokens1:
            best_match_score = 0
            best_match_token = None
            for token2 in tokens2:
                match_score = fuzzy_token_match(token1, token2)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match_token = token2
            if best_match_score > 0.8 and best_match_token not in matched_tokens:
                matched_tokens.add(best_match_token)
                overlap_score += 3 * best_match_score
        fuzzy_matching_scores.append(overlap_score / 10)

    return fuzzy_matching_scores


def get_caption_scores(query, caption_by_object_timestamp):

    captions = []
    for track_uuid, caption_by_timestamp in caption_by_object_timestamp.items():
        captions.extend(list(caption_by_timestamp.values()))

    clip_scores = clip_similarity(query, captions, batch_size=32)
    fuzzy_scores = fuzzy_matching(query, captions)

    i = 0
    caption_scores = {}
    for track_uuid, caption_by_timestamp in caption_by_object_timestamp.items():
        caption_scores[track_uuid] = {}
        for timestamp, _ in caption_by_timestamp.items():
            combined_score = clip_scores[i] + fuzzy_scores[i]
            caption_scores[timestamp] = combined_score
            i += 1

    return caption_scores

def get_best_image_crop(log_dir:Path, track_uuid, timestamp, crop_names_and_infos):

    crop_path = log_dir/f'cache/object_crops/{track_uuid}/{int(timestamp)}.png'
    if crop_path.exists():
        return (track_uuid, timestamp, Image.open(crop_path))

    best_cam = None
    best_crop = None
    best_score = -1
    for cam_name, crop_info in crop_names_and_infos:
        if crop_info['crop_area'] > best_score:
            best_score = crop_info['crop_area']
            best_cam = cam_name
            best_crop = crop_info['crop']

    image_crop = get_img_crop(best_cam, timestamp, log_dir, best_crop)
    crop_path.parent.mkdir(exist_ok=True, parents=True)
    image_crop.save(crop_path)

    return (track_uuid, timestamp, image_crop)

def get_cropped_images_by_object(log_dir, eval_timestamps=None):
    """
    Here cropped ima
    """

    all_image_crops = get_all_crops(log_dir, timestamps=eval_timestamps)
    print('All crop bboxes found.')

    reorganized_crops = {}
    for timestamp, crops_by_cam in all_image_crops.items():

        if eval_timestamps and timestamp not in eval_timestamps:
            continue

        for cam_name, crops_by_uuid in crops_by_cam.items():
            for track_uuid, crop_info in crops_by_uuid.items():

                if (track_uuid, timestamp) not in reorganized_crops:
                    reorganized_crops[(track_uuid, timestamp)] = [(cam_name, crop_info)]
                else:
                    reorganized_crops[(track_uuid, timestamp)].append((cam_name, crop_info))

    with mp.Pool(processes=32) as pool:
        object_timestamp_crop = pool.starmap(get_best_image_crop, [(log_dir, track_uuid, timestamp, crop_names_and_infos) for (track_uuid, timestamp), crop_names_and_infos in reorganized_crops.items()])

    crops_by_object_timestamp = {}
    for track_uuid, timestamp, crop in object_timestamp_crop:
        if track_uuid not in crops_by_object_timestamp:
            crops_by_object_timestamp[track_uuid] = {}
        crops_by_object_timestamp[track_uuid][timestamp] = crop

    return crops_by_object_timestamp


def get_object_motion(log_dir:Path, eval_timestamps):
    """
    Motion statistics according to ReferGPT: C_it = [x,y,z,theta,distance,dx,dy,dz,dtheta] in R^9

    Returns dict[dict[list]]:
        {track_uuid:{timestamp:[x,y,z,theta,distance,dx,dy,dz,dtheta]}}
    """
    
    cache_path = log_dir/'cache/object_motion.json'
    if cache_path.exists():
        return json.load(open(cache_path, 'rb'))

    all_uuids = get_uuids_of_category(log_dir, category="ANY")
    ego_vehicle = get_ego_uuid(log_dir)

    T = 10  # timestep horizon to caclulate dx, dy, dz, dtheta over
    motion_by_object_timestamp = {}
    for track_uuid in tqdm(all_uuids, desc='Getting motions descritions by track.'):

        positions, timestamps = get_nth_pos_deriv(
            track_uuid, 0, log_dir, coordinate_frame=ego_vehicle
        )
        velocities, _ = get_nth_pos_deriv(
            track_uuid, 1, log_dir, coordinate_frame=ego_vehicle
        )
        distances = np.linalg.norm(positions, axis=1)

        yaws, _ = get_nth_yaw_deriv(
            track_uuid, 0, log_dir, coordinate_frame=ego_vehicle, in_degrees=True
        )
        ang_vel, _ = get_nth_yaw_deriv(
            track_uuid, 1, log_dir, coordinate_frame=ego_vehicle, in_degrees=True
        )

        for i, timestamp in enumerate(timestamps):
            if timestamp not in eval_timestamps:
                continue
            if track_uuid not in motion_by_object_timestamp:
                motion_by_object_timestamp[track_uuid] = {}

            motion_by_object_timestamp[track_uuid][timestamp] = [
                positions[i, 0],
                positions[i, 1],
                positions[i, 2],
                yaws[i],
                distances[i],
                velocities[i, 0],
                velocities[i, 1],
                velocities[i, 2],
                ang_vel[i],
            ]

    cache_path.parent.mkdir(exist_ok=True)
    json.dump(motion_by_object_timestamp, open(cache_path, 'w'), indent=4)

    return motion_by_object_timestamp


def filter_tracks(scores_by_object_timestamp, threshold):

    referred_tracks = {}
    for track_uuid, scores_by_timestamp in scores_by_object_timestamp.items():
        for timestamp, score in scores_by_timestamp.items():
            if score > threshold:
                if track_uuid not in referred_tracks:
                    referred_tracks[track_uuid] = [timestamp]
                else:
                    referred_tracks[track_uuid].append(timestamp)

    return referred_tracks


def get_referred_tracks(query, log_dir):
    """
    ReferGPT Method
    """
    eval_timestamps = get_eval_timestamps(log_dir)

    motion_by_object_timestamp = get_object_motion(log_dir, eval_timestamps)
    print('Object motion accumulation complete!')

    cropped_images_by_object_timestamp = get_cropped_images_by_object(
        log_dir, eval_timestamps
    )
    print('All objects cropped!')

    # Get
    caption_by_object_timestamp = get_object_captions(
        cropped_images_by_object_timestamp,
        motion_by_object_timestamp,
        log_dir
    )
    print('All crops captioned')

    scores_by_object_timestamp = get_caption_scores(query, caption_by_object_timestamp)

    referred_tracks = filter_tracks(scores_by_object_timestamp)
    # output_scenario(referred_tracks, query, log_dir, output_dir=Path('output/examples'))

    return referred_tracks


if __name__ == "__main__":

    log_dir = Path(
        '/home/crdavids/Trinity-Sync/RefAV/output/tracker_predictions/Le3DE2D_Tracking/test/1ad57a00-cc61-3f5f-9e2a-9981a57e9856'    
    )
    get_referred_tracks("cars moving from left to right", log_dir)
