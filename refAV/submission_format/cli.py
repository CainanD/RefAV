"""Offline exporter/validator. Pickle conversion is explicit and local only."""

import argparse
import gzip
import json
from pathlib import Path

from .schema import InvalidSubmission, decode, load_predictions


def export_document(predictions):
    import numpy as np

    logs = {}
    for key, frames in predictions.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("Exporter requires (log_id, prompt) tuple keys")
        log_id, prompt = key
        converted = []
        for frame in frames:
            record, dtypes = {}, {}
            for field, value in frame.items():
                if field in {"seq_id", "ego_translation_m", "description"}:
                    continue
                if isinstance(value, np.ndarray):
                    record[field] = value.tolist()
                    if value.dtype.kind in "fiu":
                        dtypes[field] = str(value.dtype)
                elif isinstance(value, np.generic):
                    record[field] = value.item()
                else:
                    record[field] = value
            if dtypes:
                record["dtypes"] = dtypes
            converted.append(record)
        logs.setdefault(log_id, {})[prompt] = converted
    document = logs
    decode(document)
    return document


def write_predictions(predictions, output):
    document = export_document(predictions)
    output = Path(output)
    opener = gzip.open if output.name.endswith(".gz") else open
    with opener(output, "wt", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("input")
    validate.add_argument("--tracking", action="store_true", default=None)
    convert = commands.add_parser("convert-trusted-pickle", help="Never use on untrusted pickle files")
    convert.add_argument("input")
    convert.add_argument("output")
    convert.add_argument("--i-trust-this-pickle", action="store_true", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            predictions = load_predictions(args.input, tracking=args.tracking)
            print(f"Valid schema: {len(predictions)} sequences. Server also checks split coverage.")
        else:
            import pickle  # Explicit opt-in local CLI only.

            with open(args.input, "rb") as stream:
                predictions = pickle.load(stream)
            write_predictions(predictions, args.output)
            print("Converted and validated trusted predictions.")
    except (InvalidSubmission, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
