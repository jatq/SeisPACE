from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SeisPACE 1D ablation curves")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "summary_1d.csv",
    )
    parser.add_argument(
        "--proposed-json",
        type=Path,
        default=ROOT
        / "runs"
        / "seispace_ablation"
        / "eval_1d_proposed"
        / "snr_0.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation" / "figures",
    )
    return parser.parse_args()


def load_series(path: Path) -> dict[str, dict[int, float]]:
    series: dict[str, dict[int, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["snr_db"] == "mean":
                continue
            series.setdefault(row["method"], {})[int(row["snr_db"])] = float(
                row["gain_db"]
            )
    return series


def main() -> None:
    args = parse_args()
    series = load_series(args.summary)
    styles = {
        "baseline": ("1D baseline", "#555555", "--", "o"),
        "random_no_anchor": ("Random, prior", "#B07AA1", ":", "x"),
        "noisy_continuation": ("No-prior continuation", "#9C755F", "-.", "s"),
        "no_consolidation": ("No consolidation", "#F28E2B", "--", "P"),
        "warm_no_anchor": ("Warm, no anchor", "#4E79A7", "-.", "^"),
        "random_anchor": ("Random + anchor", "#E15759", ":", "v"),
        "proposed": ("SeisPACE-1D", "#D62728", "-", "D"),
    }
    prior_payload = json.loads(args.proposed_json.read_text(encoding="utf-8"))
    prior_gain = float(prior_payload["prior_metrics"]["gain_db"])

    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(5.4, 3.4))
    axis.axhline(
        prior_gain,
        color="#7F7F7F",
        linewidth=1.0,
        linestyle=(0, (1, 2)),
        label="Learned prior only",
    )
    for method, (label, color, linestyle, marker) in styles.items():
        if method not in series:
            continue
        levels = [level for level in LEVELS if level in series[method]]
        values = [series[method][level] for level in levels]
        axis.plot(
            levels,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.5,
            linewidth=1.35,
            label=label,
        )
    axis.set_xlabel("Input SNR (dB)")
    axis.set_ylabel("SNR gain (dB)")
    axis.set_xticks(LEVELS)
    axis.grid(True, linewidth=0.45, alpha=0.35)
    axis.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output / f"seispace_1d_snr.{suffix}", dpi=300)
    plt.close(fig)
    print(args.output / "seispace_1d_snr.png")


if __name__ == "__main__":
    main()
