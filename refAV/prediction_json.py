"""Leaderboard prediction JSON; tracking inputs and organizer labels keep their existing formats."""

import json

import numpy as np

ARRAYS = {
    "track_id",
    "score",
    "label",
    "name",
    "translation_m",
    "size",
    "yaw",
    "velocity_m_per_s",
}
VECTORS = {"translation_m", "size", "velocity_m_per_s"}


def write_predictions(predictions, path):
    document = {}
    for (log_id, prompt), frames in predictions.items():
        records = []
        for frame in frames:
            record = {"timestamp_ns": int(frame["timestamp_ns"])}
            dtypes = {}
            for field in ARRAYS | {"is_positive"}:
                if field not in frame:
                    continue
                value = frame[field]
                if isinstance(value, np.ndarray):
                    record[field] = value.tolist()
                    if value.dtype.kind in "fiu":
                        dtypes[field] = str(value.dtype)
                else:
                    record[field] = (
                        value.item() if isinstance(value, np.generic) else value
                    )
            if dtypes:
                record["dtypes"] = dtypes
            records.append(record)
        document.setdefault(log_id, {})[prompt] = records
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(
            document, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )


def read_predictions(path):
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    predictions = {}
    for log_id, prompts in document.items():
        for prompt, records in prompts.items():
            frames = []
            for record in records:
                frame = {"timestamp_ns": record["timestamp_ns"]}
                for field in ARRAYS & record.keys():
                    value = np.asarray(
                        record[field],
                        dtype=record.get("dtypes", {}).get(
                            field, str if field == "name" else None
                        ),
                    )
                    frame[field] = value.reshape(-1, 3) if field in VECTORS else value
                if "is_positive" in record:
                    frame["is_positive"] = record["is_positive"]
                frame["seq_id"] = (log_id, prompt)
                frames.append(frame)
            predictions[(log_id, prompt)] = frames
    return predictions
