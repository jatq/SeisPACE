from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a cached 2D section example")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="test_0")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = np.load(args.data_dir / f"{args.prefix}_clean.npy")[args.index]
    noisy = np.load(args.data_dir / f"{args.prefix}_noisy.npy")[args.index]
    prior = np.load(args.data_dir / f"{args.prefix}_prior.npy")[args.index]
    limit = float(max(np.percentile(np.abs(clean), 99), 1e-6))
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), sharey=True)
    for axis, image, title in zip(
        axes,
        (clean, noisy, prior),
        ("Reference", "Noisy", "Learned prior"),
        strict=True,
    ):
        axis.imshow(
            image,
            cmap="seismic",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=9)
        axis.set_xlabel("Trace")
    axes[0].set_ylabel("Time sample")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
