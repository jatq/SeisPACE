from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SNR gain curves from a result CSV")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    if "models" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "models"})
    snr_columns = [column for column in frame.columns if column != "models"]
    x = [float(column) for column in snr_columns]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for _, row in frame.iterrows():
        ax.plot(x, [float(row[column]) for column in snr_columns], marker="o", label=row["models"])
    ax.set_xlabel("Input SNR (dB)")
    ax.set_ylabel("SNR gain (dB)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
