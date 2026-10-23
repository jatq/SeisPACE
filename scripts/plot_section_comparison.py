from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one seismic section comparison")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["clean", "noisy", "scrn", "curvelet", "fx_decon", "wavelet"],
    )
    parser.add_argument("--title", default="Seismic section comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.input)
    available = [name for name in args.methods if name in data.files]
    if not available:
        raise ValueError("No requested methods were found in the input file")

    reference = np.asarray(data["clean"])
    scale = float(np.percentile(np.abs(reference), 99.5))
    scale = max(scale, 1e-6)
    columns = 3
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(10, 3.2 * rows), constrained_layout=True
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, name in zip(axes, available):
        image = axis.imshow(
            np.asarray(data[name]),
            aspect="auto",
            cmap="seismic",
            vmin=-scale,
            vmax=scale,
            interpolation="nearest",
        )
        axis.set_title(name)
        axis.set_xlabel("Trace")
        axis.set_ylabel("Time sample")
        figure.colorbar(image, ax=axis, shrink=0.8)
    for axis in axes[len(available) :]:
        axis.axis("off")
    figure.suptitle(args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
