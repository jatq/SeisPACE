from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a 2D anchor weight using validation histories"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Entries of the form weight=subdirectory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    return parser.parse_args()


def best_validation_snr(history_path: Path) -> float:
    with history_path.open(newline="", encoding="utf-8") as handle:
        values = [
            float(row["val_snr_db"])
            for row in csv.DictReader(handle)
            if row["val_snr_db"]
        ]
    if not values:
        raise ValueError(f"No validation values in {history_path}")
    return max(values)


def main() -> None:
    args = parse_args()
    candidates = {}
    for item in args.candidates:
        weight, tag = item.split("=", maxsplit=1)
        candidates[weight] = {
            "tag": tag,
            "validation_snr_db": best_validation_snr(
                args.root / tag / "history.csv"
            ),
        }
    selected_weight = max(
        candidates, key=lambda weight: candidates[weight]["validation_snr_db"]
    )
    selected_tag = candidates[selected_weight]["tag"]
    selected_checkpoint = args.root / selected_tag / "best.pt"
    result = {
        "selection_metric": "maximum validation SNR among anchor-weight candidates",
        "candidates": candidates,
        "selected_weight": float(selected_weight),
        "selected_tag": selected_tag,
        "checkpoint": str(selected_checkpoint),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.destination is not None:
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_checkpoint, args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
