## RefAV: Mining Referred Scenarios in Autonomous Vehicle Datasets using LLMs

This serves as the official baseline for the 2025 Argoverse2 Scenario Mining Challenge. It imitates using SQL queries on a large dataset to find scenarios of interest.

### Installation

All of the required libaries and packages can be installed with

`pip install -r requirements.txt`

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

### Running the Code

All of the code necessary for unpacking the dataset, generating referred track predictions,
and evaluating the predictions against the ground truth can be found in the `tutorial.ipynb` file.
It also includes some basic tutorials about how to define and visualize a scenario.

### Benchmark Evaluation

| **Metric** | **Description** |
|------------|-----------------|
| HOTA-Temporal | HOTA on temporally localized tracks. |
| HOTA | HOTA on the full length of a track |
| Timestamp F1 | Timestamp level classification metric |
| Scenario F1 | Scenario level classification metric. |

### Submission Format

The evaluation expects a dictionary of lists of dictionaries
```python
{
      <(log_id,prompt)>: [
            {
                  "timestamp_ns": <timestamp_ns>,
                  "track_id": <track_id>
                  "score": <score>,
                  "label": <label>,
                  "name": <name>,
                  "translation_m": <translation_m>,
                  "size": <size>,
                  "yaw": <yaw>,
            }
      ]
}
```

log_id: Log id associated with the track, also called seq_id.
prompt: The prompt/description string that describes the scenario associated with the log.
timestamp_ns: Timestamp associated with the detections.
track_id: Unique id assigned to each track, this is produced by your tracker.
score: Track confidence.
label: Integer index of the object class. This is 0 for REFERRED_OBJECTs, 1 for RELATED_OBJECTs, and 2 for OTHER_OBJECTs
name: Object class name.
translation_m: xyz-components of the object translation in the city reference frame, in meters.
size: Object extent along the x,y,z axes in meters.
yaw: Object heading rotation along the z axis.
An example looks like this:

### Example Submission
```python
example_tracks = {
  ('02678d04-cc9f-3148-9f95-1ba66347dff9','vehicle turning left'): [
    {
       'timestamp_ns': 315969904359876000,
       'translation_m': array([[6759.51786422, 1596.42662849,   57.90987307],
             [6757.01580393, 1601.80434654,   58.06088218],
             [6761.8232099 , 1591.6432147 ,   57.66341136],
             ...,
             [6735.5776378 , 1626.72694938,   59.12224152],
             [6790.59603472, 1558.0159741 ,   55.68706682],
             [6774.78130127, 1547.73853494,   56.55294184]]),
       'size': array([[4.315736  , 1.7214599 , 1.4757565 ],
             [4.3870926 , 1.7566483 , 1.4416479 ],
             [4.4788623 , 1.7604711 , 1.4735452 ],
             ...,
             [1.6218852 , 0.82648355, 1.6104599 ],
             [1.4323177 , 0.79862624, 1.5229694 ],
             [0.7979312 , 0.6317313 , 1.4602867 ]], dtype=float32),
      'yaw': array([-1.1205611 , ... , -1.1305285 , -1.1272993], dtype=float32),
      'name': array(['REGULAR_VEHICLE', ..., 'STOP_SIGN', 'REGULAR_VEHICLE'], dtype='<U31'),
      'label': array([ 0, 0, ... 9,  0], dtype=int32),
      'score': array([0.54183, ..., 0.47720736, 0.4853499], dtype=float32),
      'track_id': array([0, ... , 11, 12], dtype=int32),
    },
    ...
  ],
  ...
}
```

### Contact 

Any questions or discussion are welcome! Please raise an issue (preferred), or send me an email.

Cainan Davidson [crdavids@andrew.cmu.edu]

