import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
import os
from tqdm import tqdm
from pathlib import Path
import json
import multiprocessing as mp
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import refAV.paths as paths
from refAV.visualize import visualize_rgb

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:

    # Use standard Euclidean distance
    n = len(embeddings)

    with torch.no_grad():
        embeddings = torch.tensor(embeddings).to('cuda:1')
        distances = torch.zeros((n, n)).to('cuda:1')
        
        # Compute squared distances efficiently
        gram_matrix = embeddings @ embeddings.T
        sq_norms = torch.diag(gram_matrix)
        
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        distances = torch.sqrt(torch.maximum(
            sq_norms.unsqueeze(1) + sq_norms.unsqueeze(0) - 2 * gram_matrix,
            torch.zeros((n,n)).to('cuda:1')  # Clamp to avoid numerical errors
        ))
        
        return distances.detach().cpu().numpy()

def furthest_point_sampling(embeddings: np.ndarray) -> tuple:

    n = len(embeddings)
    
    # Compute all pairwise distances
    print("Computing pairwise distances...")
    distances = compute_pairwise_distances(embeddings)
    
    # Find the maximum distance in the dataset (excluding diagonal)
    np.fill_diagonal(distances, 0)  # Ensure diagonal is 0
    max_distance = np.max(distances)
    
    print(f"Maximum distance between any two points: {max_distance:.6f}")
    
    if max_distance == 0 or np.isnan(max_distance):
        raise ValueError("Maximum distance is 0 or NaN. Check your embeddings.")
    
    # Find the two points with maximum distance to start
    max_idx = np.unravel_index(np.argmax(distances), distances.shape)
    first_idx = max_idx[0]
    
    print(f"Starting with point {first_idx} (one end of max distance pair)")
    
    # Initialize
    ordered_indices = [first_idx]
    remaining_indices = set(range(n)) - {first_idx}
    
    # Track minimum distances from each point to the ordered set
    min_distances = distances[first_idx].copy()
    min_distances[first_idx] = 0  # Distance to itself is 0
    
    # Array to store diversity scores
    diversity_scores = np.zeros(n)
    diversity_scores[0] = 1.0  # First point gets score of 1
    
    # Perform furthest point sampling
    print("Performing furthest point sampling...")
    with tqdm(total=n-1, desc="FPS Progress") as pbar:
        for i in range(1, n):
            # Find the point that is furthest from the current ordered set
            # Only consider remaining points
            remaining_mask = np.array([idx in remaining_indices for idx in range(n)])
            masked_distances = min_distances.copy()
            masked_distances[~remaining_mask] = -np.inf
            
            next_idx = np.argmax(masked_distances)
            furthest_distance = min_distances[next_idx]
            
            # Debug: Check for issues
            if np.isnan(furthest_distance) or np.isinf(furthest_distance):
                print(f"\nWarning at iteration {i}: furthest_distance is {furthest_distance}")
                print(f"next_idx: {next_idx}")
                print(f"min_distances stats: min={np.min(min_distances)}, max={np.max(min_distances)}")
            
            # Compute diversity score: closest_distance / max_distance
            diversity_score = furthest_distance / max_distance
            diversity_scores[i] = diversity_score
            
            # Add to ordered set
            ordered_indices.append(next_idx)
            remaining_indices.remove(next_idx)
            
            # Update minimum distances
            if remaining_indices:
                new_distances = distances[next_idx]
                min_distances = np.minimum(min_distances, new_distances)
            
            pbar.update(1)
    
    return ordered_indices, diversity_scores

def load_or_compute_embeddings(input_texts, cache_path, model, tokenizer, max_length=8192, batch_size=32):

    # Create cache directory if it doesn't exist
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Try to load from cache
    if os.path.exists(cache_path):
        print(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            if len(cached_data['embeddings']) == len(input_texts):
                print(f"Loaded {len(cached_data['embeddings'])} embeddings from cache")
                return cached_data['embeddings']
            else:
                print(f"Cache size mismatch. Recomputing embeddings...")
    
    # Compute embeddings
    print(f"Computing embeddings for {len(input_texts)} texts...")
    all_embeddings = []
    
    # Process in batches
    num_batches = (len(input_texts) + batch_size - 1) // batch_size
    
    with tqdm(total=len(input_texts), desc="Generating embeddings") as pbar:
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            
            # Tokenize the batch
            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                # Keep embeddings as-is without additional normalization
                all_embeddings.append(embeddings.cpu().numpy())
            
            pbar.update(len(batch_texts))
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    # Save to cache
    print(f"Saving embeddings to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump({'embeddings': all_embeddings}, f)
    
    return all_embeddings

def dataset_language_figure():

    cache_path = 'output/embeddings_cache.pkl'
    output_ordering_path = 'output/fps_ordering_drivelm.txt'
    output_scores_path = 'output/fps_diversity_scores.npy'

    # Load input texts
    dataset_path = Path('output/other_datasets/drivelm')
    all_prompts = []


    for json_file in dataset_path.iterdir():
        with open(json_file, 'rb') as file:
            dataset = json.load(file)

        for log_id, log_info in dataset.items():
            for keyframe_id, keyframe_info in log_info['key_frames'].items():
                for questions in keyframe_info['QA'].values():
                    for question in questions:
                        all_prompts.append(question["Q"])



    input_texts = list(set(all_prompts))
    print(f"Loaded {len(input_texts)} texts")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
    model = AutoModel.from_pretrained(
        'Qwen/Qwen3-Embedding-8B',
        attn_implementation="flash_attention_2",
        dtype=torch.float16,
        device_map = 'cuda:0'
    ).cuda()
    model.eval()
    
    # Load or compute embeddings
    embeddings = load_or_compute_embeddings(
        input_texts,
        cache_path,
        model,
        tokenizer,
        max_length=8192,
        batch_size=32
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Perform furthest point sampling
    ordered_indices, diversity_scores = furthest_point_sampling(embeddings)
    
    # Save the ordering
    print(f"Saving FPS ordering to: {output_ordering_path}")
    with open(output_ordering_path, 'w') as f:
        for idx, (orig_idx, score) in enumerate(zip(ordered_indices, diversity_scores)):
            f.write(f"{input_texts[orig_idx]} {score:.4f}\n")
    
    # Save diversity scores as numpy array
    print(f"Saving diversity scores to: {output_scores_path}")
    np.save(output_scores_path, diversity_scores)
    
    print("\nSummary:")
    print(f"Total data points: {len(input_texts)}")
    print(f"First point index: {ordered_indices[0]} (diversity score: {diversity_scores[0]:.6f})")
    print(f"Last point index: {ordered_indices[-1]} (diversity score: {diversity_scores[-1]:.6f})")
    print(f"Mean diversity score: {np.mean(diversity_scores):.6f}")
    print(f"Median diversity score: {np.median(diversity_scores):.6f}")
    
    print("\nDone!")

def scenario_distribution_figure():

    df = pd.read_feather('scenario_mining_downloads/dataset_all_splits.feather')
    # Create a boolean mask for non-OTHER_OBJECT categories
    df['is_positive'] = df['mining_category'] != 'OTHER_OBJECT'
    
    # Group by prompt and log_id, check if any row in each log_id group is positive
    grouped = df.groupby(['prompt', 'log_id'])['is_positive'].any().reset_index()
    
    # Now group by prompt and count positive/negative log_ids
    scenario_counts = grouped.groupby('prompt')['is_positive'].agg([
        ('positive', 'sum'),
        ('negative', lambda x: (~x).sum())
    ]).to_dict('index')
    
    # Convert to the desired format
    scenario_counts = {
        prompt: {'positive': int(counts['positive']), 'negative': int(counts['negative'])}
        for prompt, counts in scenario_counts.items()
    }

    with open('output/scenario_counts.json', 'w') as file:
        json.dump(scenario_counts, file, indent=4)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    with open('output/scenario_counts.json', 'rb') as file:
        data = json.load(file)

    num_positive = 0
    num_negative = 0
    for scenario, counts in data.items():
        num_positive += counts['positive']
        num_negative += counts['negative']

    num_scenarios = 10
    step = len(data.keys())//num_scenarios
    ordered_scenarios = sorted(data.keys(), key=lambda k:(-data[k]['positive']))
    kept_scenarios = []
    kept_positives = []
    kept_negatives = []
    for i in range(0, len(ordered_scenarios), step):
        scenario = ordered_scenarios[i]
        kept_scenarios.append(scenario)
        kept_positives.append(data[scenario]['positive'])
        kept_negatives.append(data[scenario]['negative'])

    sorted_indices = sorted(np.arange(len(kept_scenarios)), key=lambda i:(-kept_positives[i], -kept_negatives[i]))
    kept_scenarios = np.array(kept_scenarios)[sorted_indices]
    kept_positives = np.array(kept_positives)[sorted_indices]
    kept_negatives = np.array(kept_negatives)[sorted_indices]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))  # Changed 'axes' to 'ax'

    # ============================================================================
    # 1. Basic Bar Chart with Custom Labels
    # ============================================================================
    # First bar: positives (green, at bottom)
    bars1 = ax.bar(kept_scenarios, kept_positives, color="#55CA59", label='Positive Match')

    # Second bar: negatives (red, stacked on top using bottom parameter)
    bars2 = ax.bar(kept_scenarios, kept_negatives, bottom=kept_positives, color="#F05454", label='Negative Match')

    ax.set_title('Occurences within RefAV by Scenario', fontsize= 16, fontweight='bold')
    ax.set_xlabel('Scenario Description', fontsize=12)
    ax.set_ylabel('Number of Occurrences', fontsize=12)
    ax.legend(fontsize=12)  # Add legend to show which color is which

    for bar, pos_val in zip(bars1, kept_positives):
        if pos_val > 0:  # Only show non-zero values
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{int(pos_val)}',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    for bar, neg_val, pos_val in zip(bars2, kept_negatives, kept_positives):
        if neg_val > 0:  # Only show non-zero values
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., pos_val + height/2,
                    f'{int(neg_val)}',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.tight_layout()
    plt.savefig('output/visualization/scenario_distribution.png')

def teaser_figure():

    dataset_dir = paths.AV2_DATA_DIR
    feather_path = Path(
        "output/visualization/b40c0cbf-5d35-30df-9f63-de088ada278e/turning left_b40c0cbf_annotations.feather"
    )
    output_dir = Path("output/visualization")
    log_id = "b40c0cbf-5d35-30df-9f63-de088ada278e"

    visualize_rgb(
        dataset_dir,
        feather_path,
        output_dir,
        log_id,
        description="Vehicle making left turn through ego-vehicle's path while it is raining",
    )

if __name__ == "__main__":
    teaser_figure()

