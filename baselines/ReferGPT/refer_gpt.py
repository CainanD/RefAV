from openai import OpenAI
import pandas as pd
import numpy as np
import time
import traceback
from transformers import (
    AutoModel,
    AutoProcessor,
    Siglip2Model,
    Siglip2Processor,
    Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration,
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
import torch.multiprocessing as tmp
import argparse
import pickle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from av2.evaluation.tracking.eval import filter_max_dist
from refAV.utils import (
    get_all_crops,
    get_nth_pos_deriv,
    get_nth_yaw_deriv,
    get_ego_uuid,
    get_uuids_of_category,
    get_eval_timestamps,
    get_img_crop,
)
import refAV.paths as paths
from refAV.atomic_functions import output_scenario
from refAV.code_generation import predict_scenario_qwen
from refAV.eval import (
    compute_temporal_metrics,
    combine_matching_pkls,
    evaluate,
    combine_pkls,
    evaluate_pkls,
)

client = OpenAI()


def clip_similarity(
    query: str,
    captions: list[str | Image.Image],
    batch_size=32,
    cache_dir=Path("output/cache"),
    gpu_id=0,
):

    # Cache checking code
    cache = {}
    query_cache = {}
    cache_path = cache_dir / "clip_similarity.json"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as file:
                cache = json.load(file)
            if query in cache:
                query_cache = cache[query]
        except:
            traceback.print_exc()
            cache_path.unlink(missing_ok=True)

    caption_indices = []
    captions_to_embed = []
    similarity_scores = np.zeros(len(captions))
    for i, caption in enumerate(captions):
        if caption in query_cache:
            similarity_scores[i] = query_cache[caption]
        else:
            caption_indices.append(i)
            captions_to_embed.append(caption)

    # If everything is cached, return early without loading model
    if not captions_to_embed:
        print("All CLIP embeddings already cached.")
        return similarity_scores

    # Get an available GPU
    device = f"cuda:{gpu_id}"

    model_id = "google/siglip2-so400m-patch16-naflex"
    siglip: Siglip2Model = AutoModel.from_pretrained(model_id, device_map=device)
    processor: Siglip2Processor = AutoProcessor.from_pretrained(model_id)

    with torch.no_grad():
        iter = 0
        texts = [query] + captions_to_embed
        text_features = []
        while iter < len(texts):
            print(f"{iter}/{len(texts)}", end="\r")
            batch_texts = texts[iter : min(len(texts), iter + batch_size)]
            inputs = processor(text=batch_texts, return_tensors="pt").to(siglip.device)
            text_features.append(siglip.get_text_features(**inputs))
            iter += batch_size

        text_features = torch.concat(text_features, axis=0)

        query_features = text_features[0].unsqueeze(0) / torch.linalg.norm(
            text_features[0]
        )
        caption_features = text_features[1:, :] / torch.linalg.norm(
            text_features[1:, :], axis=1, ord=2, keepdim=True
        )

        clip_similarity = query_features @ caption_features.T
        clip_similarity = clip_similarity.cpu().numpy()[0]

    for i in range(len(clip_similarity)):
        query_cache[captions[caption_indices[i]]] = float(clip_similarity[i])
        similarity_scores[caption_indices[i]] = clip_similarity[i]

    # Clean up
    del siglip, processor, inputs, text_features, query_features, caption_features
    torch.cuda.empty_cache()

    cache[query] = query_cache
    cache_path.parent.mkdir(exist_ok=True, parents=True)

    with open(cache_path, "w") as file:
        json.dump(cache, file, indent=4)

    return similarity_scores


def load_huggingface_model_and_processor(model_id, device):

    # Load model on this specific GPU
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map=device
    )
    model.eval()

    processor = Qwen3VLProcessor.from_pretrained(
        model_id
    )  # Memory errors @ 1600 tokens on 11GB 2080
    processor.tokenizer.padding_side = "left"

    return model, processor


def caption_worker(gpu_id, input_queue, output_queue, model_id):
    """Worker process that runs on a single GPU"""

    device = f"cuda:{gpu_id}"
    model, processor = load_huggingface_model_and_processor(model_id, device)
    print(f"GPU {gpu_id}: Model loaded and ready")

    # Process batches from queue
    while True:
        item = input_queue.get()

        if item is None:  # Poison pill to stop worker
            break

        batch_idx, batch, batch_info = item

        try:
            # Process batch
            batched_captions = generate_response(batch, model, processor)
            torch.cuda.empty_cache()

            for caption, (track_uuid, timestamp, caption_path) in zip(
                batched_captions, batch_info
            ):
                # Write to file
                caption_path.parent.mkdir(exist_ok=True, parents=True)
                with open(caption_path, "w") as f:
                    f.write(caption)

                # Print with GPU info
                print(f"(GPU {gpu_id}, {track_uuid}, {timestamp}): {caption}")

            # Send results back
            output_queue.put((batch_idx, batched_captions, batch_info))

        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR in batch {batch_idx}: {e}")
            output_queue.put((batch_idx, None, batch_info))

    print(f"GPU {gpu_id}: Shutting down")


def generate_response(message_batch, model, processor) -> list[str]:

    inputs = processor.apply_chat_template(
        message_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    print(inputs["input_ids"].shape)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    batched_response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return batched_response


# Function to create a file with the Files API
def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id


def oai_caption(
    track_uuid,
    timestamp,
    cropped_image,
    category,
    prompt_template,
    motion_by_object_timestamp,
    log_dir,
):
    caption_path: Path = log_dir / f"cache/captions/{track_uuid}/{str(timestamp)}.txt"

    if caption_path.exists():
        with open(caption_path, "r") as file:
            caption = file.read().strip()
        return caption
    else:

        motion_array = motion_by_object_timestamp[track_uuid][str(timestamp)]
        motion_string = "[" + ", ".join(f"{x:.1f}" for x in motion_array) + "]"

        try:
            file_id = create_file(cropped_image)
            start_time = time.time()
            response = client.responses.create(
                model="gpt-5-mini",
                reasoning={"effort": "minimal"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": prompt_template.format(
                                    object_motion=motion_string, category=category
                                ),
                            },
                            {
                                "type": "input_image",
                                "file_id": file_id,
                            },
                        ],
                    }
                ],
            )
            end_time = time.time()

            caption = response.output_text
            print(
                f"Captioned {track_uuid}, {category}, {timestamp} in {end_time-start_time} seconds."
            )
            print(caption)

            caption = caption
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            with open(caption_path, "w") as f:
                f.write(caption)

            return caption
        except:
            traceback.print_exc()
            return None


def get_object_captions(
    cropped_images_by_object_timestamp,
    motion_by_object_timestamp,
    log_dir,
    llm_prompt_path="run/llm_prompting/ReferGPT/caption_prompt.txt",
    batch_size=4,
):
    with open(llm_prompt_path, "r") as file:
        prompt_template = file.read()

    df = pd.read_feather(log_dir / "sm_annotations.feather")

    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    # print(f"Using {num_gpus} GPUs")

    captions_by_object_timestamp = {}
    cur_batch = []
    message_batches = []
    batch_info_list = []  # Track which captions belong to which batch

    # Prepare all batches
    for track_uuid, cropped_images_by_timestamp in tqdm(
        cropped_images_by_object_timestamp.items(),
        desc="Captioning objects with ChatGPT",
    ):
        category = df[df["track_uuid"] == track_uuid]["category"].unique()[0]
        if track_uuid not in captions_by_object_timestamp:
            captions_by_object_timestamp[track_uuid] = {}

        for timestamp, cropped_image in cropped_images_by_timestamp.items():
            cur_batch.append((track_uuid, timestamp, cropped_image, category))

    # with mp.Pool(20) as pool:
    #    captions = pool.starmap(oai_caption, [[track_uuid, timestamp, cropped_image, category, prompt_template, motion_by_object_timestamp]
    #                                          for track_uuid, timestamp, cropped_image, category in cur_batch])
    captions = []
    for track_uuid, timestamp, cropped_image, category in cur_batch:
        caption = oai_caption(
            track_uuid,
            timestamp,
            cropped_image,
            category,
            prompt_template,
            motion_by_object_timestamp,
            log_dir,
        )
        captions.append(caption)

    for (track_uuid, timestamp, _, _), caption in zip(cur_batch, captions):
        if caption is None:
            caption = "Error"

        captions_by_object_timestamp[track_uuid][timestamp] = caption

        """
                print(f'{track_uuid}, {category}, {timestamp}')

                #cropped_image:Image.Image = cropped_image
                H, W = cropped_image.size
                image_pixels = H*W
                max_pixels = 16*16*512

                if image_pixels > max_pixels:

                    scale = max_pixels / image_pixels
                    new_w = round(W*scale)
                    new_h = round(H*scale)
                    cropped_image = cropped_image.resize((new_w, new_h))

                motion_array = motion_by_object_timestamp[track_uuid][str(timestamp)]
                motion_string = '[' + ', '.join(f'{x:.1f}' for x in motion_array) + ']'

                chat = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": cropped_image
                            },
                            {
                                "type": "text",
                                "text": prompt_template.format(
                                    object_motion=motion_string, category=category
                                ),
                            },
                        ],
                    }
                ]
                
                cur_batch.append(chat)
                batch_info_list.append((track_uuid, timestamp, caption_path))
                
                if len(cur_batch) >= batch_size:
                    message_batches.append(deepcopy(cur_batch))
                    cur_batch = []
    
    # Add remaining batch
    if cur_batch:
        message_batches.append(cur_batch)
    
    print(f'Number of caption batches: {len(message_batches)}')
    
    if not message_batches:
        return captions_by_object_timestamp

    print(response.output_text)

    # Using local LLM to caption
    # Set up multiprocessing
    mp.set_start_method('spawn', force=True)
    input_queue = tmp.Queue(maxsize=num_gpus * 2)
    output_queue = tmp.Queue()
    
    # Start worker processes
    processes = []
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=caption_worker,
            args=(gpu_id, input_queue, output_queue, model_id)
        )
        p.start()
        processes.append(p)
    
    # Distribute batches to workers
    info_start_idx = 0
    for batch_idx, batch in enumerate(message_batches):
        batch_size_actual = len(batch)
        batch_info = batch_info_list[info_start_idx:info_start_idx + batch_size_actual]
        input_queue.put((batch_idx, batch, batch_info))
        info_start_idx += batch_size_actual
    
    # Send poison pills to stop workers
    for _ in range(num_gpus):
        input_queue.put(None)
    
    # Collect results with progress bar
    results = {}
    with tqdm(total=len(message_batches), desc='Generating captions') as pbar:
        for _ in range(len(message_batches)):
            batch_idx, decoded_text, batch_info = output_queue.get()
            results[batch_idx] = (decoded_text, batch_info)
            pbar.update(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Process results in order
    for batch_idx in sorted(results.keys()):
        decoded_text, batch_info = results[batch_idx]
        
        if decoded_text is None:
            print(f"Batch {batch_idx} failed")
            continue
            
        for caption, (track_uuid, timestamp, caption_path) in zip(decoded_text, batch_info):
            captions_by_object_timestamp[track_uuid][timestamp] = caption """

    return captions_by_object_timestamp


def fuzzy_matching(
    query: str, captions: list[str], cache_dir=Path(f"output/cache")
) -> list[float]:
    """Calculate token match score with fuzzy matching."""

    cache_path = cache_dir / "fuzzy_similarity.json"

    cache = {}
    query_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as file:
                cache = json.load(file)

            if query in cache:
                query_cache = cache[query]
        except:
            traceback.print_exc()
            cache_path.unlink(missing_ok=True)

    fuzzy_matching_scores = []
    for prompt in tqdm(captions, desc="Fuzzy scoring"):

        if prompt in query_cache:
            fuzzy_matching_scores.append(query_cache[prompt])
            continue

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

        score = overlap_score / 10
        query_cache[prompt] = score
        fuzzy_matching_scores.append(score)

    cache[query] = query_cache

    with open(cache_dir / "fuzzy_similarity.json", "w") as file:
        json.dump(cache, file, indent=4)

    return fuzzy_matching_scores


def get_caption_scores(query, caption_by_object_timestamp, cache_dir=None, gpu_id=0):

    captions = []
    for track_uuid, caption_by_timestamp in caption_by_object_timestamp.items():
        captions.extend(list(caption_by_timestamp.values()))

    fuzzy_scores = fuzzy_matching(query, captions, cache_dir=cache_dir)
    clip_scores = clip_similarity(
        query, captions, batch_size=64, cache_dir=cache_dir, gpu_id=gpu_id
    )

    i = 0
    caption_scores = {}
    for track_uuid, caption_by_timestamp in caption_by_object_timestamp.items():
        caption_scores[track_uuid] = {}
        for timestamp, _ in caption_by_timestamp.items():
            combined_score = clip_scores[i] + fuzzy_scores[i]/10
            caption_scores[track_uuid][timestamp] = combined_score
            i += 1

    return caption_scores


def get_best_image_crop(log_dir: Path, track_uuid, timestamp, crop_names_and_infos):

    crop_path = log_dir / f"cache/object_crops/{track_uuid}/{str(timestamp)}.png"
    if crop_path.exists():
        return (track_uuid, timestamp, crop_path)

    best_cam = None
    best_crop = None
    best_score = -1
    for cam_name, crop_info in crop_names_and_infos:
        if crop_info["crop_area"] > best_score:
            best_score = crop_info["crop_area"]
            best_cam = cam_name
            best_crop = crop_info

    H = crop_info["cam_H"]
    W = crop_info["cam_W"]

    x1, y1, x2, y2 = best_crop["crop"]
    pad_x = 0.1 * (x2 - x1)
    pad_y = 0.1 * (y2 - y1)
    x1 = max(0, x1 - pad_x)
    x2 = min(W, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(H, y2 + pad_y)

    image_crop = get_img_crop(best_cam, int(timestamp), log_dir, (x1, y1, x2, y2))
    crop_path.parent.mkdir(exist_ok=True, parents=True)
    image_crop.save(crop_path)

    return (track_uuid, timestamp, crop_path)


def get_cropped_images_by_object(log_dir, eval_timestamps):

    all_image_crops = get_all_crops(log_dir, timestamps=eval_timestamps)

    reorganized_crops = {}
    for timestamp, crops_by_cam in all_image_crops.items():

        if eval_timestamps and int(timestamp) not in eval_timestamps:
            continue

        for cam_name, crops_by_uuid in crops_by_cam.items():
            for track_uuid, crop_info in crops_by_uuid.items():

                if (track_uuid, timestamp) not in reorganized_crops:
                    reorganized_crops[(track_uuid, timestamp)] = [(cam_name, crop_info)]
                else:
                    reorganized_crops[(track_uuid, timestamp)].append(
                        (cam_name, crop_info)
                    )

    print(f"{len(reorganized_crops)} crops found.")
    """
    # Branch for initial processesing
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        object_timestamp_crop = pool.starmap(get_best_image_crop, [(log_dir, track_uuid, str(timestamp), crop_names_and_infos) for (track_uuid, timestamp), crop_names_and_infos in reorganized_crops.items()])
    print(f'Retreived {len(object_timestamp_crop)} image crops.')

    crops_by_object_timestamp = {}
    for track_uuid, timestamp, crop in object_timestamp_crop:

        if track_uuid not in crops_by_object_timestamp:
            crops_by_object_timestamp[track_uuid] = {}
        crops_by_object_timestamp[track_uuid][str(timestamp)] = crop 

    """
    # Branch for processing when all crops are already cached
    crops_by_object_timestamp = {}
    for (track_uuid, timestamp), crop_names_and_infos in reorganized_crops.items():
        if track_uuid not in crops_by_object_timestamp:
            crops_by_object_timestamp[track_uuid] = {}

        crop_path = log_dir / f"cache/object_crops/{track_uuid}/{str(timestamp)}.png"
        if crop_path.exists():
            crops_by_object_timestamp[track_uuid][str(timestamp)] = crop_path
        else:
            crop = get_best_image_crop(
                log_dir, track_uuid, timestamp, crop_names_and_infos
            )
            crops_by_object_timestamp[track_uuid][str(timestamp)] = crop

    return crops_by_object_timestamp


def get_object_motion(log_dir: Path, eval_timestamps):
    """
    Motion statistics according to ReferGPT: C_it = [x,y,z,theta,distance,dx,dy,dz,dtheta] in R^9

    Returns dict[dict[list]]:
        {track_uuid:{timestamp:[x,y,z,theta,distance,dx,dy,dz,dtheta]}}
    """

    cache_path = log_dir / "cache/object_motion.json"
    if cache_path.exists():
        with open(cache_path, "rb") as file:
            object_motion = json.load(file)
        return object_motion

    all_uuids = get_uuids_of_category(log_dir, category="ANY")
    ego_vehicle = get_ego_uuid(log_dir)

    # T = 10 # timestep horizon to caclulate dx, dy, dz, dtheta over
    # using 1 second instead of set number of steps so that dx, dy, dz are speed in m/s
    motion_by_object_timestamp = {}
    for track_uuid in tqdm(all_uuids, desc="Generating motion descriptions"):

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

            # Removed z-variation as Argoverse2 does not have much elevation change
            motion_by_object_timestamp[track_uuid][str(timestamp)] = [
                positions[i, 0],
                positions[i, 1],
                yaws[i],
                velocities[i, 0],
                velocities[i, 1],
                ang_vel[i],
                distances[i],
            ]

    cache_path.parent.mkdir(exist_ok=True)

    with open(cache_path, "w") as file:
        json.dump(motion_by_object_timestamp, file, indent=4)

    return motion_by_object_timestamp


def get_query_category(query, model=None, processor=None):

    cache = {}
    cache_path = Path("output/cache/description_category.json")
    if cache_path.exists():

        with open(cache_path, "rb") as file:
            cache = json.load(file)
        if query in cache:
            return cache[query]

    with open("run/llm_prompting/RefAV/categories.txt", "r") as file:
        categories = file.read()
    with open("run/llm_prompting/query_class_prompt.txt", "r") as file:
        prompt = file.read()
        prompt = prompt.format(query=query, categories=categories)

    if model is None:
        model, processor = load_huggingface_model_and_processor(
            "Qwen/Qwen3-VL-4B-Instruct", device="auto"
        )
    message_batch = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}]]

    category = generate_response(message_batch, model, processor)[0]
    cache[query] = category

    print(f"Predicted category for query <{query}>: {category}")
    with open(cache_path, "w") as file:
        json.dump(cache, file, indent=4)

    return category


def filter_tracks(
    scores_by_object_timestamp, query_category, threshold, log_dir, output_file=None
):
    referred_tracks = {}

    # Prepare data for visualization
    all_timestamps = []
    all_scores = []
    all_track_ids = []

    category_uuids = get_uuids_of_category(log_dir, category=query_category)

    for track_uuid, scores_by_timestamp in scores_by_object_timestamp.items():
        if track_uuid not in category_uuids:
            continue

        scores = np.array(list(scores_by_timestamp.values()))
        timestamps = list(scores_by_timestamp.keys())

        # Collect data for plotting
        all_timestamps.extend(timestamps)
        all_scores.extend(scores)
        all_track_ids.extend([track_uuid] * len(scores))

        #for timestamp, score in scores_by_timestamp.items():
        #    if score > threshold:
        #        if track_uuid not in referred_tracks:
        #            referred_tracks[track_uuid] = []
        #        referred_tracks[track_uuid].append(timestamp)

         #Majority voting
        if np.sum(scores > threshold) > len(scores)/2:
            referred_tracks[track_uuid] = [int(timestamp) for timestamp in scores_by_timestamp.keys()]

    if output_file:
        # Create visualization
        plt.figure()

        # Get unique track IDs and assign colors
        unique_tracks = list(scores_by_object_timestamp.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_tracks)))
        track_to_color = {track: colors[i] for i, track in enumerate(unique_tracks)}

        # Plot scores for each track
        for track_uuid in unique_tracks:
            track_mask = np.array(all_track_ids) == track_uuid
            track_scores = np.array(all_scores)[track_mask]
            track_timestamps = np.array(all_timestamps)[track_mask]

            plt.scatter(
                track_timestamps,
                track_scores,
                c=[track_to_color[track_uuid]],
                label=f"Track {track_uuid}",
                alpha=0.6,
                s=50,
            )

        # Add threshold line
        plt.axhline(
            y=threshold,
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )
        plt.xlabel("Timestamp")
        plt.ylabel("Score")
        plt.title("Scores by Track and Timestamp")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)
        plt.close()

    return referred_tracks


def get_referred_tracks(query, log_dir, gpu_id=0):
    """
    ReferGPT Method
    """
    print(f"Finding referred tracks for log_id {log_dir.stem} and query {query}")

    cache_dir = log_dir / "cache"
    eval_timestamps = get_eval_timestamps(log_dir)

    motion_by_object_timestamp = get_object_motion(log_dir, eval_timestamps)
    print("Object motion accumulation complete!")

    cropped_images_by_object_timestamp = get_cropped_images_by_object(
        log_dir, eval_timestamps
    )

    print(
        f"Getting captions for {len(list(cropped_images_by_object_timestamp.keys()))} objects"
    )
    caption_by_object_timestamp = get_object_captions(
        cropped_images_by_object_timestamp,
        motion_by_object_timestamp,
        log_dir,
        batch_size=2,
    )
    print("All crops captioned")

    query_category = get_query_category(query)

    scores_by_object_timestamp = get_caption_scores(
        query, caption_by_object_timestamp, cache_dir=cache_dir, gpu_id=gpu_id
    )

    figure_path = Path(
        f"baselines/ReferGPT/scenario_predictions/{log_dir.parent.stem}/{log_dir.stem}/{query[:30]}_scores.png"
    )
    referred_tracks = filter_tracks(
        scores_by_object_timestamp,
        query_category,
        threshold=0.63,
        log_dir=log_dir,
        # output_file=figure_path,
    )

    return referred_tracks


def process_query_log_pair(query, log_dir):

    # Get the current worker's ID and map it to a GPU
    worker_id = mp.current_process()._identity[0] - 1  # 0-indexed
    gpu_id = worker_id % torch.cuda.device_count()

    output_dir = Path(f"baselines/ReferGPT/scenario_predictions/test/Le3DE2D_Tracking")
    referred_tracks = get_referred_tracks(query, log_dir, gpu_id=gpu_id)
    output_scenario(
        referred_tracks, query, log_dir, output_dir=output_dir, visualize=False
    )


if __name__ == "__main__":

    split = "test"

    parser = argparse.ArgumentParser(description="Example script with arguments")
    parser.add_argument(
        "--log_prompt_pairs",
        type=str,
        default=f"scenario_mining_downloads/log_prompt_pairs_{split}.json",
    )
    parser.add_argument("--start_log", type=int, required=True)
    parser.add_argument("--end_log", type=int, required=True)
    args = parser.parse_args()

    pred_base_dir = Path("output/tracker_predictions/Le3DE2D_Tracking")
    split = Path(args.log_prompt_pairs).stem.split(sep="_")[-1]
    output_dir = Path(
        f"baselines/ReferGPT/scenario_predictions/{split}/{pred_base_dir.stem}"
    )

    with open(args.log_prompt_pairs, "rb") as file:
        log_prompt_pairs = json.load(file)
    calibration_logs = list(log_prompt_pairs.keys())[args.start_log : args.end_log]
    query_logs_to_process = []
    log_ids_to_combine = []

    for log_id in calibration_logs:

        log_ids_to_combine.append(log_id)
        log_dir = pred_base_dir / split / log_id

        for i, query in enumerate(log_prompt_pairs[log_id]):
            """
            #if i > 0:
            #    break

            #Sequential computation branch
            referred_tracks = get_referred_tracks(query, log_dir)
            output_scenario(referred_tracks, query, log_dir, output_dir=output_dir, visualize=False)
            """
            pkl_path = (
                output_dir
                / "scenario_predictions"
                / log_id
                / f"{query}_predictions.pkl"
            )
            if pkl_path.exists():
                continue

            # Branch for when all information is already cached
            query_logs_to_process.append((query, log_dir))

    print(f"Processing {len(query_logs_to_process)} query-log pairs")
    mp.set_start_method("spawn", force=True)
    num_gpus = torch.cuda.device_count()
    with mp.Pool(num_gpus) as pool:
        pool.starmap(
            process_query_log_pair,
            [(query, log_dir) for query, log_dir in query_logs_to_process],
            chunksize=1,
        )

    combined_preds = combine_pkls(output_dir, Path(args.log_prompt_pairs))
    combined_gt = Path(
        f"../RefAV-Construction/output/eval/{split}/latest/combined_gt_{split}.pkl"
    )
    metrics = evaluate_pkls(combined_preds, combined_gt, output_dir)
    print(metrics)

    """
    combine_matching_pkls(
        paths.SM_DATA_DIR / split,
        output_dir,
        output_dir=output_dir,
        method_name="ReferGPT",
        log_ids_to_combine=log_ids_to_combine,
    ) 

    with open(output_dir / "combined_gt.pkl", "rb") as file:
        ground_truth = pickle.load(file)
        filter_max_dist(ground_truth, 50)
    with open(output_dir / "ReferGPT_predictions.pkl", "rb") as file:
        predictions = pickle.load(file)
        filter_max_dist(predictions, 50)

    log_ba, timestamp_ba = compute_temporal_metrics(
        predictions, ground_truth, str(output_dir)
    )
    print(f"Log balanced accuracy: {log_ba}")
    print(f"Timestamp balanced accuracy: {timestamp_ba}")"""
