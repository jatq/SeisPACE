from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert server wavelet MAT files to the upstream NPZ format"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def convert(input_path: Path, output_path: Path) -> None:
    data = loadmat(str(input_path))
    x = data["R"].T.astype(np.float32)
    n = data["z"].T.astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, x=x, n=n)
    print(f"{input_path} -> {output_path} x={x.shape} n={n.shape}")


def main() -> None:
    args = parse_args()
    convert(
        args.input_root / "training_data.mat",
        args.output_root / "training_data.npz",
    )
    convert(
        args.input_root / "testing_data.mat",
        args.output_root / "testing_data.npz",
    )


if __name__ == "__main__":
    main()
