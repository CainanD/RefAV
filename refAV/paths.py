from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('/data3/shared/datasets/ArgoVerse2/Sensor')
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('scenario_mining_downloads')

# input directories, do not change
EXPERIMENTS = Path('run/experiment_configs/experiments.yml')
PROMPT_DIR = Path('refAV/llm_prompting')

# output directories, do not change
SM_DATA_DIR = Path('output/sm_dataset')
SM_PRED_DIR = Path('output/sm_predictions')
LLM_PRED_DIR = Path('output/llm_code_predictions/NuPrompt')
TRACKER_PRED_DIR = Path('output/tracker_predictions')

# path to cached atomic function outputs, likely does not exist for you
CACHE_PATH = Path('output/cache')
