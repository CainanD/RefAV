import gzip
import json
import math
import re
from pathlib import Path
from uuid import UUID

JSON_TOKENS = re.compile(rb'\\.|["{}\[\]\\]', re.DOTALL)

MAX_JSON_BYTES = 4 * 1024**3
CATEGORIES = ("REFERRED_OBJECT", "RELATED_OBJECT", "OTHER_OBJECT")
FLOAT_FIELDS = {"score": 1, "translation_m": 3, "size": 3, "yaw": 1, "velocity_m_per_s": 3}
TRACK_FIELDS = {"track_id", "score", "label", "name", "translation_m", "size", "yaw"}
DTYPES = {"float32", "float64", "int32", "int64"}


class InvalidSubmission(ValueError):
    """Only deliberately sanitized messages may be displayed to participants."""


def fail(message):
    raise InvalidSubmission(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("Duplicate JSON object keys are not allowed.")
        result[key] = value
    return result


def reject_constant(value):
    fail("NaN and Infinity are not allowed.")


def read_document(path, max_bytes=MAX_JSON_BYTES):
    """Bound decompressed bytes and nesting before invoking the JSON decoder."""
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open
    try:
        data = bytearray()
        depth, quoted, escaped = 0, False, False
        with opener(path, "rb") as stream:
            while chunk := stream.read(min(1024 * 1024, max_bytes - len(data) + 1)):
                data.extend(chunk)
                if len(data) > max_bytes:
                    fail("Decompressed JSON exceeds the size limit.")
                # Match only structural tokens and complete escape pairs in C,
                # rather than scanning every numeric byte in Python. A trailing
                # string escape consumes the first byte of the following chunk.
                scan = chunk
                if escaped:
                    scan = scan[1:]
                    escaped = False
                for match in JSON_TOKENS.finditer(scan):
                    token = match[0]
                    if quoted:
                        if token[0] == 92:
                            escaped = len(token) == 1
                        elif token == b'"':
                            quoted = False
                    elif token == b'"':
                        quoted = True
                    elif token in (b"[", b"{"):
                        depth += 1
                        if depth > 16:
                            fail("JSON nesting exceeds the limit.")
                    elif token in (b"]", b"}"):
                        depth -= 1
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, EOFError, RecursionError, ValueError) as exc:
        if isinstance(exc, InvalidSubmission):
            raise
        raise InvalidSubmission("The upload is not valid UTF-8 JSON or gzip JSON.") from exc


def integer(value, minimum=0, maximum=2**63 - 1):
    return type(value) is int and minimum <= value <= maximum


def numeric(value):
    return type(value) in (int, float) and abs(value) < 1e30 and math.isfinite(value)


def decode(document, tracking=None):
    """Validate primitives before constructing a limited set of NumPy arrays."""
    import numpy as np

    logs = document
    if not isinstance(logs, dict) or not logs or len(logs) > 1000:
        fail("Expected a nonempty top-level object of log IDs.")
    sequences, mode, total_frames = {}, None, 0
    for log_id, prompts in logs.items():
        try:
            if str(UUID(log_id)) != log_id:
                fail("Log IDs must be canonical UUID strings.")
        except (ValueError, TypeError, AttributeError):
            fail("Log IDs must be canonical UUID strings.")
        if not isinstance(prompts, dict) or not prompts or len(prompts) > 10000:
            fail("Each log must contain a nonempty object of prompts.")
        for prompt, frames in prompts.items():
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 4096:
                fail("Prompts must contain between 1 and 4096 characters.")
            if not isinstance(frames, list) or not frames or len(frames) > 10000:
                fail("Each prompt must contain a nonempty frame list.")
            total_frames += len(frames)
            if total_frames > 2_000_000:
                fail("Too many frames.")
            timestamps, converted = set(), []
            for frame in frames:
                if not isinstance(frame, dict) or not integer(frame.get("timestamp_ns")):
                    fail("Every frame needs a nonnegative int64 timestamp_ns.")
                timestamp = frame["timestamp_ns"]
                if timestamp in timestamps:
                    fail("Duplicate timestamps are not allowed within a sequence.")
                timestamps.add(timestamp)
                is_tracking = "track_id" in frame
                if mode is not None and mode != is_tracking:
                    fail("Do not mix temporal-only and tracking frames.")
                mode = is_tracking
                if tracking is True and not is_tracking:
                    fail("This phase requires tracking predictions.")
                if not is_tracking:
                    if (
                        set(frame) != {"timestamp_ns", "is_positive"}
                        or type(frame.get("is_positive")) is not bool
                    ):
                        fail("Temporal frames require only timestamp_ns and boolean is_positive.")
                    converted.append(dict(frame))
                    continue
                allowed = TRACK_FIELDS | {"timestamp_ns", "dtypes", "velocity_m_per_s", "is_positive"}
                if not TRACK_FIELDS.issubset(frame) or not set(frame).issubset(allowed):
                    fail("Tracking frame fields do not match the submission format.")
                if "is_positive" in frame and type(frame["is_positive"]) is not bool:
                    fail("is_positive must be a boolean.")
                dtypes = frame.get("dtypes", {})
                if not isinstance(dtypes, dict) or not set(dtypes).issubset(
                    FLOAT_FIELDS.keys() | {"track_id", "label"}
                ):
                    fail("Invalid dtype declarations.")
                for field, dtype in dtypes.items():
                    if not isinstance(dtype, str) or dtype not in DTYPES:
                        fail("Only float32, float64, int32, and int64 dtypes are supported.")
                    if (field in FLOAT_FIELDS) != dtype.startswith("float"):
                        fail("dtype does not match its field.")
                ids = frame["track_id"]
                if not isinstance(ids, list) or len(ids) > 10000 or any(not integer(x) for x in ids):
                    fail("track_id must be a list of nonnegative int64 values (at most 10000).")
                if len(ids) != len(set(ids)):
                    fail("Track IDs must be unique within a frame.")
                count, result = len(ids), {"timestamp_ns": timestamp}
                for field in TRACK_FIELDS | ({"velocity_m_per_s"} if "velocity_m_per_s" in frame else set()):
                    values = frame[field]
                    if not isinstance(values, list) or len(values) != count:
                        fail("All tracking arrays must have the same length.")
                    if field == "name":
                        if any(type(x) is not str or x not in CATEGORIES for x in values):
                            fail("Invalid tracking category names.")
                        result[field] = np.asarray(values, dtype="<U31")
                    elif field in {"label", "track_id"}:
                        maximum = 2 if field == "label" else 2**63 - 1
                        if any(not integer(x, maximum=maximum) for x in values):
                            fail("Invalid integer tracking values.")
                        dtype = dtypes.get(field, "int64")
                        if dtype == "int32" and any(x > 2**31 - 1 for x in values):
                            fail("Integer value exceeds declared dtype.")
                        result[field] = np.asarray(values, dtype=dtype)
                    else:
                        width = FLOAT_FIELDS[field]
                        if width == 1:
                            valid = all(numeric(x) for x in values)
                        else:
                            valid = all(
                                isinstance(x, list) and len(x) == width and all(numeric(v) for v in x)
                                for x in values
                            )
                        if not valid:
                            fail(
                                "Tracking coordinates and scores must be finite numeric arrays with valid shapes."
                            )
                        result[field] = np.asarray(values, dtype=dtypes.get(field, "float64"))
                        if width > 1:
                            result[field] = result[field].reshape(count, width)
                        if field == "size" and np.any(result[field] < 0):
                            fail("Object dimensions must be nonnegative.")
                if any(CATEGORIES[label] != name for label, name in zip(frame["label"], frame["name"])):
                    fail("Tracking labels and category names disagree.")
                if "is_positive" in frame:
                    result["is_positive"] = frame["is_positive"]
                converted.append(result)
            sequences[(log_id, prompt)] = sorted(converted, key=lambda x: x["timestamp_ns"])
    return sequences


def load_predictions(path, tracking=None, max_bytes=MAX_JSON_BYTES):
    return decode(read_document(path, max_bytes), tracking)


def align_to_ground_truth(predictions, ground_truth):
    """Reject partial submissions and reorder by timestamps, not list position."""
    if set(predictions) != set(ground_truth):
        fail("Prediction sequences do not match the required split coverage.")
    aligned = {}
    for key, gt_frames in ground_truth.items():
        frames = {frame["timestamp_ns"]: frame for frame in predictions[key]}
        expected = [int(frame["timestamp_ns"]) for frame in gt_frames]
        if len(expected) != len(set(expected)) or set(frames) != set(expected):
            fail("Prediction timestamps do not match the required split coverage.")
        aligned[key] = [frames[timestamp] for timestamp in expected]
    return aligned
