from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("m10", "0", "p10")
LEVEL_LABELS = ("-10 dB", "0 dB", "10 dB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SeisPACE 2D ablations")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "summary_2d.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "figures",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["method"], {})[row["snr_db"]] = float(
                row["achieved_snr_db"]
            )
    return result


def grouped_bars(
    axis: plt.Axes,
    values: dict[str, dict[str, float]],
    methods: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
) -> None:
    x = np.arange(len(LEVELS))
    width = 0.13
    visible = [method for method in methods if method in values]
    offsets = (np.arange(len(visible)) - (len(visible) - 1) / 2) * width
    for offset, method in zip(offsets, visible, strict=True):
        heights = [values[method].get(level, np.nan) for level in LEVELS]
        axis.bar(
            x + offset,
            heights,
            width,
            label=labels[method],
            color=colors[method],
            edgecolor="white",
            linewidth=0.5,
        )
    axis.set_xticks(x, LEVEL_LABELS)
    axis.set_ylabel("Achieved SNR (dB)")
    axis.grid(axis="y", linewidth=0.45, alpha=0.35)


def main() -> None:
    args = parse_args()
    values = load_summary(args.summary)
    colors = {
        "baseline": "#555555",
        "random_no_anchor": "#B07AA1",
        "noisy_continuation": "#9C755F",
        "no_consolidation": "#F28E2B",
        "warm_no_anchor": "#4E79A7",
        "random_anchor": "#E15759",
        "proposed": "#D62728",
        "prior_zero": "#76B7B2",
        "prior_shuffled": "#EDC948",
    }
    labels = {
        "baseline": "SCRN baseline",
        "random_no_anchor": "Random, prior",
        "noisy_continuation": "No-prior continuation",
        "no_consolidation": "No consolidation",
        "warm_no_anchor": "Warm, no anchor",
        "random_anchor": "Random + anchor",
        "proposed": "SeisPACE-2D",
        "prior_zero": "Zero prior",
        "prior_shuffled": "Shuffled prior",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axis = plt.subplots(figsize=(6.0, 3.4))
    grouped_bars(
        axis,
        values,
        [
            "baseline",
            "random_no_anchor",
            "noisy_continuation",
            "no_consolidation",
            "warm_no_anchor",
            "random_anchor",
            "proposed",
        ],
        colors,
        labels,
    )
    axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seispace_2d_ablation.{suffix}", dpi=300)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(4.4, 3.3))
    grouped_bars(
        axis,
        values,
        ["proposed", "prior_zero", "prior_shuffled"],
        colors,
        labels,
    )
    axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seispace_2d_prior_control.{suffix}", dpi=300)
    plt.close(fig)
    print(args.output / "seispace_2d_ablation.png")


if __name__ == "__main__":
    main()
