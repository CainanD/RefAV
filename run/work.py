from refAV.eval import combine_pkls, evaluate_pkls
from pathlib import Path

experiment_dir = Path('/home/crdavids/Trinity-Sync/refbot/output/sm_predictions/exp1')
lpp_path = Path('/home/crdavids/Trinity-Sync/refbot/av2_sm_downloads/log_prompt_pairs_test.json')
combined_preds = combine_pkls(experiment_dir, lpp_path)
combined_gt = Path('/home/crdavids/Trinity-Sync/av2-api/output/eval/test/latest/combined_gt_test.pkl')

metrics = evaluate_pkls(combined_preds, combined_gt, experiment_dir)
print(metrics)