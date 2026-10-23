from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot anchor-weight selection")
    parser.add_argument(
        "--selection",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "main_selection.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.9), sharey=False)
    for axis, key, title in (
        (axes[0], "one_dimensional", "1D validation"),
        (axes[1], "two_dimensional", "2D validation"),
    ):
        values = selection[key]["anchor_validation_snr_db"]
        labels = list(values)
        heights = [float(values[label]) for label in labels]
        axis.bar(labels, heights, color="#4E79A7", width=0.58)
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Anchor weight")
        axis.grid(axis="y", linewidth=0.4, alpha=0.3)
        for index, value in enumerate(heights):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Validation SNR (dB)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seispace_anchor_weight.{suffix}", dpi=300)
    plt.close(fig)
    print(args.output / "seispace_anchor_weight.png")


if __name__ == "__main__":
    main()
