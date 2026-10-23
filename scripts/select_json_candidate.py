from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select candidate JSON by metric")
    parser.add_argument(
        "--candidates",
        nargs="+",
        required=True,
        help="Entries of the form tag=path",
    )
    parser.add_argument("--metric", default="metrics.snr_db")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested_value(payload: dict, path: str) -> float:
    value = payload
    for key in path.split("."):
        value = value[key]
    return float(value)


def main() -> None:
    args = parse_args()
    candidates = {}
    for item in args.candidates:
        tag, path_text = item.split("=", maxsplit=1)
        path = Path(path_text)
        candidates[tag] = {
            "path": str(path),
            "metric": nested_value(
                json.loads(path.read_text(encoding="utf-8")), args.metric
            ),
        }
    selected = max(candidates, key=lambda tag: candidates[tag]["metric"])
    result = {
        "metric": args.metric,
        "selected": selected,
        "selected_path": candidates[selected]["path"],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(selected)


if __name__ == "__main__":
    main()
