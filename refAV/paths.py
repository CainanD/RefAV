from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('/data3/shared/datasets/ArgoVerse2/Sensor')
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('scenario_mining_downloads')

# Required to run evaluation on nuPrompt/nuScenes dataset, ignore otherwise
NUPROMPT_DATA_DIR = Path('/data/crdavids/nuscenes/nuscenes/nuprompt_v1.0')
NUSCENES_DIR = Path('/ssd0/nperi/nuScenes/v1.0-trainval')
NUSCENES_AV2_DATA_DIR = Path('/home/crdavids/Trinity-Sync/RefAV/output/tracker_predictions/PFTrack_Tracking')

# input directories, do not change
EXPERIMENTS = Path('run/experiment_configs/experiments.yml')
PROMPT_DIR = Path('run/llm_prompting')

# output directories, do not change
SM_DATA_DIR = Path('output/sm_dataset')
SM_PRED_DIR = Path('output/sm_predictions')
LLM_PRED_DIR = Path('output/llm_code_predictions')
TRACKER_PRED_DIR = Path('output/tracker_predictions')
CACHE_PATH = Path('output/cache')
