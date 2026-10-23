from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SeisPACE 1D seed results")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "seed_summary_1d.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["rows"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axis = plt.subplots(figsize=(3.6, 3.2))
    x = [0, 1]
    for row in rows:
        axis.plot(
            x,
            [row["baseline_mean_gain_db"], row["seispace_mean_gain_db"]],
            color="#9A9A9A",
            linewidth=0.8,
            alpha=0.8,
        )
    axis.scatter(
        [0] * len(rows),
        [row["baseline_mean_gain_db"] for row in rows],
        color="#555555",
        s=28,
        label="Direct baseline",
        zorder=3,
    )
    axis.scatter(
        [1] * len(rows),
        [row["seispace_mean_gain_db"] for row in rows],
        color="#D62728",
        s=28,
        label="SeisPACE-1D",
        zorder=3,
    )
    axis.set_xticks(x, ["Baseline", "SeisPACE"])
    axis.set_ylabel("Mean SNR gain (dB)")
    axis.grid(axis="y", linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seispace_1d_seeds.{suffix}", dpi=300)
    plt.close(fig)
    print(args.output / "seispace_1d_seeds.png")


if __name__ == "__main__":
    main()
