from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("m10", "0", "p10")
LABELS = ("-10 dB", "0 dB", "10 dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize native 2D field results")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "runs" / "seispace_field",
    )
    parser.add_argument("--baseline-dir", default="eval_forge2d_baseline")
    parser.add_argument("--no-prior-dir", default="eval_forge2d_consolidated")
    parser.add_argument(
        "--no-consolidation-dir", default="eval_forge2d_no_consolidation"
    )
    parser.add_argument("--seispace-dir", default="eval_forge2d_seispace")
    parser.add_argument("--prior-dir")
    parser.add_argument("--output-prefix", default="forge2d_summary")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    methods = {
        "baseline": [
            args.root / args.baseline_dir / f"{level}.json"
            for level in LEVELS
        ],
        "no_prior": [
            args.root / args.no_prior_dir / f"{level}.json"
            for level in LEVELS
        ],
        "no_consolidation": [
            args.root / args.no_consolidation_dir / f"{level}.json"
            for level in LEVELS
        ],
        "seispace": [
            args.root / args.seispace_dir / f"{level}.json"
            for level in LEVELS
        ],
    }
    summary = {"methods": {}}
    for method, paths in methods.items():
        if not all(path.exists() for path in paths):
            continue
        summary["methods"][method] = {
            level: read_json(path)["metrics"]
            for level, path in zip(LEVELS, paths, strict=True)
        }
    if args.prior_dir is not None:
        prior_root = args.root / args.prior_dir
        prior_paths = [prior_root / f"{level}.json" for level in LEVELS]
    else:
        prior_paths = [
            args.root / f"prior_field_{level}.json" for level in LEVELS
        ]
    if all(path.exists() for path in prior_paths):
        summary["prior_only"] = {
            level: read_json(path)["prior"]
            for level, path in zip(LEVELS, prior_paths, strict=True)
        }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    available = summary["methods"]
    if available:
        fig, axis = plt.subplots(figsize=(5.6, 3.2))
        x = np.arange(len(LEVELS))
        has_prior = int("prior_only" in summary)
        total_bars = len(available) + has_prior
        width = 0.8 / total_bars
        labels = {
            "baseline": "SCRN baseline",
            "no_prior": "Consolidated, no prior",
            "no_consolidation": "No consolidation",
            "seispace": "SeisPACE-2D",
        }
        colors = {
            "baseline": "#555555",
            "no_prior": "#9C755F",
            "no_consolidation": "#F28E2B",
            "seispace": "#D62728",
        }
        for index, (method, values) in enumerate(available.items()):
            axis.bar(
                x + (index - (total_bars - 1) / 2) * width,
                [values[level]["snr_db"] for level in LEVELS],
                width,
                label=labels[method],
                color=colors[method],
            )
        if "prior_only" in summary:
            axis.bar(
                x + (len(available) - (total_bars - 1) / 2) * width,
                [summary["prior_only"][level]["snr_db"] for level in LEVELS],
                width,
                label="Retrained prior only",
                color="#4E79A7",
            )
        axis.set_xticks(x, LABELS)
        axis.set_ylabel("Achieved SNR (dB)")
        axis.grid(axis="y", linewidth=0.4, alpha=0.3)
        axis.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(
                args.root / f"{args.output_prefix}.{suffix}", dpi=300
            )
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
