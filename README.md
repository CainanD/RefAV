## RefAV: Mining Referred Scenarios in Autonomous Vehicle Datasets using LLMs

<p align="center">
  <img src="figures/pipeline.png" alt="RefAV Method">
</p>

A single autonomous vehicle will stream about ~4TB of data per hour with a full stack of camera and lidar sensors. The vast majority of this data comes from uninteresting scenarios -- the ego vehicle driving straight down a lane, possibly with another car in front of it. It can be prohibitively expensive to retrive and label specific scenarios for ego-behaivor evaluation, safety testing, or active learning at scale.

RefAV serves as the baseline for the 2025 Argoverse2 Scenario Mining Challenge. It utilizes an LLM to construct composable function calls from a set of hand-crafted atomic functions such as "turning" or "has_objects_in_front". Given a prompt, the LLM outputs a composable function that narrows down a set of bounding box track predictions to the set that best corresponds to the prompt. Our paper describing the dataset and baseline in detail is available on [arXiv](https://arxiv.org/pdf/2505.20981)  

The ongoing [leaderboard](https://scenario-mining.argoverse.org/) supports versioned results. Past prize competitions are closed.

### Dataset versions

| Version | Dataset | Evaluation split |
|---|---|---|
| V1 (2025) | [Frozen HF revision](https://huggingface.co/datasets/CainanD/RefAV/tree/a2a8e2348661408f304ab205c62ff3ff3dcf194e) | Original 2025 scenarios |
| V2 (2026) | [Frozen HF revision](https://huggingface.co/datasets/CainanD/RefAV/tree/3bc556d1d8b6b52484596d3b097675d3684c5074) | Original 2026 scenarios |
| V3 | V2 training files unchanged; validation/test restricted to `generation_type == "manual"` | 307 validation and 326 test scenarios in the verified local source |

V3 keeps scenario identities, timestamps, and splits. Procedural scenarios are excluded only from validation and test, including procedural annotations subsequently reviewed by humans. The release manifest records the immutable HF dataset revision and RefAV code revision; publication is pending until those artifacts are verified.

V3 accepts tracking and track-free submissions. Results rank by **HOTA-Temporal**, then **Timestamp BA**, then submission date and ID. Track-free submissions receive zero HOTA-Temporal and HOTA-Track; their temporal balanced accuracies are evaluated normally. Tracking remains necessary for nonzero HOTA scores.

The manual evaluation subset and track-free option reduce evaluation work and address organizer-reported reliability concerns about generated 3D tracks. Historical datasets and scores remain available.

Submissions are hidden by default. Public display requires both the submitter's opt-in and administrator approval; submitters can always hide their own results. Approval controls public display, not ordinary submission or evaluation.

### Installation

Using [Conda](https://anaconda.org/anaconda/conda) is recommended for environment management
```
conda create -n refAV python=3.10
conda activate refAV
```

All of the required libaries and packages can be installed with

```
pip install -r requirements.txt
export PYTHONPATH=.
```

Running this code requires downloading the Argoverse2 test and val splits. Run the commands below to download the entire sensor dataset.
More information can be found in the [Argoverse User Guide](https://argoverse.github.io/user-guide/getting_started.html#downloading-the-data).
```
conda install s5cmd -c conda-forge

export DATASET_NAME="sensor"  # sensor, lidar, motion_forecasting or tbv.
export TARGET_DIR="$HOME/data/datasets"  # Target directory on your machine.

s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/$DATASET_NAME/*" $TARGET_DIR
```
It also requies downloading the scenario-mining add on. 
```
export TARGET_DIR="$(pwd)/av2_sm_downloads"
s5cmd --no-sign-request cp "s3://argoverse/tasks/scenario_mining/*" $TARGET_DIR
```

### Generating Detections and Tracks
See the [LT3D repository](https://github.com/neeharperi/LT3D) for information on training a baseline detector and tracker on the Argoverse 2 dataset.

We provide additional tracking outputs from previous winning Argoverse submissions on the test set [here](https://drive.google.com/file/d/1X19D5pBBO56eb_kvPOePLLhHDCsY0yql/view)

### Running the Code

All of the code necessary for unpacking the dataset, generating referred track predictions,
and evaluating the predictions against the ground truth can be found in the `tutorial.ipynb` file.
It also includes some basic tutorials about how to define and visualize a scenario.

Our experimental results and test/val submissions can be reproduced directly by running `python run_experiment.py --exp_name <exp_name>`. All experiments are found in the `experiments.yml` file.

### Benchmark Evaluation

| **Metric** | **Description** |
|------------|-----------------|
| HOTA-Temporal | HOTA on temporally localized tracks |
| HOTA-Track | HOTA on the full length of a track |
| Timestamp Balanced Accuracy | Timestamp level classification metric |
| Log Balanced Accuracy | Data log/scenario level classification metric |

### Submission Format

Upload `.json` or `.json.gz` to the [leaderboard](https://scenario-mining.argoverse.org/). Pickle is an internal cache format and is not accepted by the website.

```json
{
  "02678d04-cc9f-3148-9f95-1ba66347dff9": {
    "a car turning left": [
      {"timestamp_ns": 315969904359876000, "is_positive": true}
    ]
  }
}
```

Include every required log/prompt and exact timestamp, including negative frames. There is no wrapper. Keep nanosecond timestamps as exact integers. Tracking frames instead contain equal-length `track_id`, `score`, `label`, `name`, `translation_m`, `size`, and `yaw` arrays; optional `dtypes` preserves numeric precision. Do not include ego poses; the server supplies trusted poses. Mixed tracking/temporal frames and incomplete submissions are rejected.

```bash
PYTHONPATH=. python run/run_experiment.py --exp_name YOUR_EXPERIMENT \
  --dataset-root /path/to/v3 --sensor-root /path/to/av2 \
  --submission-format json.gz --predictions-only
python -m refAV.submission_format.cli validate /path/to/predictions.json.gz
```

Add `--temporal-only` to export timestamp booleans. The RefAV baseline may still use tracks internally. This projection means a timestamp is positive when the baseline marks it positive or predicts a referred object; it does not reproduce historical tracking-derived BA after confidence tuning. For local tracking evaluation, omit `--predictions-only` and pass `--ground-truth /path/to/trusted/annotations.pkl`. The experiment name must exist in `run/experiments.yml` and use available model/tracker inputs.

### Leaderboards

- [Current V3 leaderboard](https://scenario-mining.argoverse.org/)
- [2026 leaderboard (V2)](https://scenario-mining.argoverse.org/versions/v2/)
- [2025 leaderboard (V1)](https://scenario-mining.argoverse.org/versions/v1/)

### Additional Competition Details

* Language queries are object-centric -- all correspond to some set of objects.
* Most language queries are given from the third person persective (such as "ego vehicle turning left"). The language queries given from the first-person perspective (such as "the pedestrian on the right") describe objects from the point of view of the ego vehicle.
* In the case the language query does not refer to an object (such as "raining"), the track bounding boxes should be drawn around the ego vehicle.
* Scenarios only involve objects within 50 meters from the ego vehicle and within 5 meters of a mapped road.
* Interacting objects within a scenario are at most 50 meters away from each other. 
* All referred object tracks persist for at least 3 evaluation timestamps (1.5s).
  
The ego vehicle has the following bounding box across all logs and timestamps
'translation_m': [1.422, 0, 0.25]
'size': [4.877, 2, 1.473]
'yaw': [0]

### Contact 

Any questions or discussion are welcome! Please raise an issue (preferred), or send me an email.

Cainan Davidson [crdavids@andrew.cmu.edu]

## Citation
If you find our paper and code repository useful, please cite us:
```bib
@article{davidson2025refav,
  title={RefAV: Towards Planning-Centric Scenario Mining},
  author={Davidson, Cainan and Ramanan, Deva and Peri, Neehar},
  journal={arXiv preprint arXiv:2505.20981},
  year={2025}
}
```
