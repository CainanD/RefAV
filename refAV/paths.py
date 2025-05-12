from pathlib import Path

# change to path where the Argoverse2 Sensor dataset is downloaded
AV2_DATA_DIR = Path('/data3/shared/datasets/ArgoVerse2/Sensor')
TRACKER_DOWNLOAD_DIR = Path('tracker_downloads')
SM_DOWNLOAD_DIR = Path('av2_sm_downloads')

# path to cached atomic function outputs, likely does not exist for you
CACHE_PATH = Path('/home/crdavids/Trinity-Sync/av2-api/output/misc')

#input directories, do not change
EXPERIMENTS = Path('run/experiments.yml')
REFAV_CONTEXT = Path('refAV/llm_prompting/refAV_context.txt')
AV2_CATEGORIES = Path('refAV/llm_prompting/av2_categories.txt')
PREDICTION_EXAMPLES = Path('refAV/llm_prompting/prediction_examples.txt')

#output directories, do not change
SM_DATA_DIR = Path('output/sm_dataset')
SM_PRED_DIR = Path('output/sm_predictions')
LLM_PRED_DIR = Path('output/llm_code_predictions')
TRACKER_PRED_DIR = Path('output/tracker_predictions')

