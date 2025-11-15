from openai import OpenAI
import anthropic
from pathlib import Path
import os
import json
import pandas as pd
import multiprocessing as mp
import time

from refAV.paths import AV2_DATA_DIR, SM_PRED_DIR
from refAV.atomic_functions import output_scenario
from refAV.eval import combine_pkls, evaluate_pkls
from refAV.utils import get_log_split

def process_log_predictions(description, log_id,  scenario_file, experiment_dir, tracker_dir):

    tracker_log_dir = tracker_dir / log_id
    pkl_path = experiment_dir / 'scenario_predictions' / log_id / f'{description}_predictions.pkl'
    if pkl_path.exists():
        return
    try:
        scenario_df = pd.read_csv(scenario_file)
        scenario = (
            scenario_df.groupby("track_uuid")["timestamp_ns"].apply(list).to_dict()
        )
        scenario = {str(k): v for k, v in scenario.items()}
    except:
        scenario = {}
    #print(scenario)

    output_scenario(scenario, description, tracker_log_dir, experiment_dir / 'scenario_predictions')    

def evaluate_generated_scenarios(
    log_prompts_path, llm_output_dir: Path, tracker_dir:Path, experiment_dir: Path
):
    """
    Args:
        log_prompt_path: path to the log_prompt_pair json file
        llm_output_dir: Path to the parent of the log_dirs containing the LLM-generated scenario_i.csv files
        tracker_dir: Path to the parent of the log_dirs container the sm_annotations.feather files for each tracker
        experiment_dir: Path to save the experiment results to
    """

    with open(log_prompts_path, "rb") as file:
        log_prompt_pairs = json.load(file)

    files_to_process = []
    for log_dir in llm_output_dir.iterdir():

        split = get_log_split(log_dir)
        log_id = log_dir.stem
        prompts = log_prompt_pairs[log_id]

        lowest_index = None
        log_scenario_files = []
        for scenario_file in log_dir.iterdir():
            if "scenario" in scenario_file.name:
                # Scenarios are names scenario_1.csv -> scenario_n.csv
                prompt_index = int(scenario_file.stem.split(sep="_")[1])
                log_scenario_files.append(scenario_file)
                if lowest_index is None:
                    lowest_index = prompt_index
                else:
                    lowest_index = min(lowest_index, prompt_index)

        for scenario_file in log_scenario_files:
            prompt_index = int(scenario_file.stem.split(sep="_")[1]) - lowest_index
            description = prompts[prompt_index]
            files_to_process.append((description, log_id, scenario_file))

        if len(files_to_process) < len(prompts):
            print(f"Missing {len(prompts) - len(files_to_process)} prompt(s) in log {log_id}")

    with mp.Pool(mp.cpu_count()-1) as pool:
        pool.starmap(process_log_predictions, [(description, log_id, scenario_file, experiment_dir, tracker_dir) for description, log_id, scenario_file in files_to_process])

    combined_preds = combine_pkls(experiment_dir, log_prompts_path)
    combined_gt = Path(
        f"../RefAV-Construction/output/eval/{split}/latest/combined_gt_{split}.pkl"
    )
    metrics = evaluate_pkls(combined_preds, combined_gt, experiment_dir)
    print(metrics)


def mine_scenarios_open_ai(log_id, prompts, tracker_path):

    split = "test"
    log_dir = tracker_path / log_id
    output_path = Path(f"output/black_box/{log_id}")

    if output_path.exists():
        for file in output_path.iterdir():
            if "scenario" in file.name:
                print(f"Cached scenario found for log {log_id}")
                return f"Cached scenario found for log {log_id}"

    client = OpenAI(api_key=str(os.environ["RefAV_OPENAI_API_KEY"]))
    container = client.containers.create(name=log_id)

    annotations_csv_path = log_dir / "sm_annotations.csv"
    df = pd.read_feather(log_dir / "sm_annotations.feather")
    df.to_csv(annotations_csv_path, index=False)
    with open(annotations_csv_path, "rb") as file:
        client.containers.files.create(
            container_id=container.id,
            file=file,
        )

    map_dir = Path(AV2_DATA_DIR / split / log_id / "map")
    for filepath in map_dir.iterdir():
        if "map" in filepath.name:
            map_filepath = filepath
    with open(map_filepath, "rb") as file:
        client.containers.files.create(
            container_id=container.id,
            file=file,
        )

    ego_poses_path = AV2_DATA_DIR / split / log_id / "city_SE3_egovehicle.feather"
    ego_csv_path = log_dir / "city_SE3_egovehicle.csv"
    df = pd.read_feather(ego_poses_path)
    df.to_csv(ego_csv_path, index=False)
    with open(ego_csv_path, "rb") as file:
        client.containers.files.create(
            container_id=container.id,
            file=file,
        )

    with open("run/llm_prompting/BlackBox/prompt.txt", "r") as file:
        instructions = file.read()

    # Now use the container with the file in your response
    print(f"Sending files for log {log_id} to GPT-5")
    resp = client.responses.create(
        model="gpt-5",
        tools=[{"type": "code_interpreter", "container": container.id}],
        tool_choice="required",
        instructions=instructions,
        input=f"Identify if each of these scenarios occurs within the log.\n\n{prompts}",
    )
    time.sleep(2)
    output_files = client.containers.files.list(container_id=container.id)

    output_path.mkdir(exist_ok=True, parents=True)
    for file in output_files:
        if file.path.endswith(".csv"):
            file_content = client.containers.files.content.retrieve(
                container_id=container.id, file_id=file.id
            )

            # Read as bytes and decode to string
            content_bytes = file_content.read()
            content_text = content_bytes.decode("utf-8")

            # Write as text
            with open(output_path / Path(file.path).name, "w", encoding="utf-8") as f:
                f.write(content_text)

            print("File saved as human-readable text!")

    with open(output_path / "response.txt", "w") as file:
        file.write(str(resp))

    return resp.output_text


def mine_scenarios_anthropic(log_id, prompts, tracker_path):

    # Currently does not work
    log_dir = tracker_path / log_id
    output_path = Path(f"output/black_box/claude-sonnet-4-5/{log_id}")

    if output_path.exists():
        for file in output_path.iterdir():
            if "scenario" in file.name:
                print(f"Cached scenario found for log {log_id}")
                return f"Cached scenario found for log {log_id}"

    client = anthropic.Anthropic()

    annotations_csv_path = log_dir / "sm_annotations.csv"
    df = pd.read_feather(log_dir / "sm_annotations.feather")
    df.to_csv(annotations_csv_path, index=False)
    with open(annotations_csv_path, "rb") as file:
        annotations = client.beta.files.upload(
            file=file,
        )

    map_dir = Path(AV2_DATA_DIR / split / log_id / "map")
    for filepath in map_dir.iterdir():
        if "map" in filepath.name:
            map_filepath = filepath
    with open(map_filepath, "rb") as file:
        map = client.beta.files.upload(
            file=file,
        )

    ego_poses_path = AV2_DATA_DIR / split / log_id / "city_SE3_egovehicle.feather"
    ego_csv_path = log_dir / "city_SE3_egovehicle.csv"
    df = pd.read_feather(ego_poses_path)
    df.to_csv(ego_csv_path, index=False)
    with open(ego_csv_path, "rb") as file:
        ego_poses = client.beta.files.upload(
            file=file,
        )

    with open("run/llm_prompting/BlackBox/prompt.txt", "r") as file:
        instructions = file.read()

    # Now use the container with the file in your response
    print(f"Sending files for log {log_id} to Claude-4-5")
   # Accumulate stream events
    content_blocks = []
    full_text = ""
    
    with client.beta.messages.stream(
        model="claude-sonnet-4-5",
        max_tokens=64000,
        betas=["code-execution-2025-08-25", "files-api-2025-04-14"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": instructions + f"Identify if each of these scenarios occurs within the log.\n\n{prompts}"},
                {"type": "container_upload", "file_id": annotations.id},
                {"type": "container_upload", "file_id": ego_poses.id},
                {"type": "container_upload", "file_id": map.id}
            ]
        }],
        tools=[{
            "type": "code_execution_20250825",
            "name": "code_execution"
        }]
    ) as stream:
        for event in stream:
            # Accumulate text
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    full_text += event.delta.text
            
            # Store content blocks
            if event.type == "content_block_stop":
                content_blocks.append(event.content_block)
        
        # Get the final message
        final_message = stream.get_final_message()
    
    # Extract file IDs from the final message
    file_ids = []
    for block in final_message.content:
        if block.type == 'tool_use':
            # Check if there are output files in the tool use result
            if hasattr(block, 'output') and block.output:
                # For code execution results
                if isinstance(block.output, dict):
                    if 'files' in block.output:
                        for file_info in block.output['files']:
                            if 'file_id' in file_info:
                                file_ids.append(file_info['file_id'])
        
        # Alternative: check content for file references
        if hasattr(block, 'content') and isinstance(block.content, list):
            for content_item in block.content:
                if hasattr(content_item, 'file_id'):
                    file_ids.append(content_item.file_id)
    
    # Create output directory
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Download the created files
    print(f"Found {len(file_ids)} output files")
    for file_id in file_ids:
        try:
            file_metadata = client.beta.files.retrieve_metadata(file_id)
            file_content = client.beta.files.download(file_id)
            
            output_file_path = output_path / file_metadata.filename
            file_content.write_to_file(output_file_path)
            print(f"Downloaded: {file_metadata.filename}")
        except Exception as e:
            print(f"Error downloading file {file_id}: {e}")
    
    # Save the full text response
    with open(output_path / "response.txt", "w") as file:
        file.write(full_text)
    
    return full_text


if __name__ == "__main__":

    split = "test"
    tracker_path = Path("output/tracker_predictions/Le3DE2D_Tracking") / split
    log_prompts_path = Path(f"scenario_mining_downloads/log_prompt_pairs_{split}.json")
    experiment_dir = SM_PRED_DIR / "claude-sonnet-4-5"

    with open(log_prompts_path, "rb") as file:
        lpp_superset = json.load(file)
        lpp = []
        for i, (log_id, prompts) in enumerate(lpp_superset.items()):
            if i > 0:
                break

            lpp.append((log_id, prompts))

    with mp.Pool(20) as pool:
        responses = pool.starmap(
            mine_scenarios_anthropic, [(log_id, prompts, tracker_path) for log_id, prompts in lpp]
        )

    output_path = Path(f"output/black_box/claude-sonnet-4-5")
    evaluate_generated_scenarios(log_prompts_path, output_path, tracker_path, experiment_dir)
