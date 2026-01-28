## Evaluating on the NuPrompt Dataset

[NuPrompt](https://github.com/wudongming97/Prompt4Driving) is a referring multi-object tracking (RMOT) dataset where the task is predict 
sequences of 3D bounding boxes from a given text prompt. It is build on top of the NuScenes dataset.

### Download and Convert NuScenes to the Argoverse2 format.

Download the [NuScenes][https://www.nuscenes.org/nuscenes] dataset. 
`python run/nuprompt_conversion/nuscenes_to_av2.py`

### Run 3D tracker on NuScenes and convert to the Argoverse2 format.

We evaluate with two camera-only trackers: [PF-Track](https://github.com/TRI-ML/PF-Track) and [StreamPETR](https://github.com/exiawsh/StreamPETR). Follow the instructions in the respective repos to produce a tracking_results.json file. 

After obtaining the tracking results run
`python run/nuprompt_conversion/nuscenes_to_av2.py`

### Run RefProg 

Modify your `run/experiment_configs/experiments.yml` file to include an experiment with your NuScenes tracking results and NuPrompt descriptions.

`python run/run_experiment.py --exp exp62`

### Convert to RefAV format to NuPrompt format and Evaluate

`python run/nuprompt_conversion/refav_to_nuprompt.py`

Finally, run evaluation in the [NuPrompt](https://github.com/CainanD/Prompt4Driving-AV2) repo.
`python tools/eval_results_file.py`