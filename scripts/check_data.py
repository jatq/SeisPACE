from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.config import load_config  # noqa: E402
from seispace.transforms import ist_dct, st_dct  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate data paths and transforms")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    return parser.parse_args()


def check_wavelet(config) -> None:
    from scipy.io import loadmat

    for name in [config.data.train_file, config.data.test_file]:
        path = Path(config.data.root) / name
        if not path.exists():
            raise FileNotFoundError(path)
        data = loadmat(str(path))
        print(name, {key: data[key].shape for key in ["R", "z"] if key in data})
        if "R" not in data or "z" not in data:
            raise KeyError(f"{path} must contain R and z")


def check_stead(config) -> None:
    required = [
        config.data.train_hdf5,
        config.data.train_meta,
        config.data.noise_hdf5,
        config.data.test_hdf5,
        config.data.test_meta,
    ]
    for name in required:
        path = Path(config.data.root) / name
        if not path.exists():
            raise FileNotFoundError(path)
        print(name, path.stat().st_size)
    try:
        import h5py
    except ImportError:
        print("h5py is not installed in this environment; HDF5 contents were not opened")
        return
    for name in [config.data.train_hdf5, config.data.noise_hdf5, config.data.test_hdf5]:
        with h5py.File(Path(config.data.root) / name, "r") as handle:
            keys = list(handle["data"].keys())[:3]
            print(name, "keys", keys, "sample_shape", handle["data"][keys[0]].shape)


def check_transform(config) -> None:
    rng = np.random.default_rng(0)
    for length in [300, 2048, 6000]:
        signal = rng.normal(size=length).astype(np.float32)
        recovered = ist_dct(st_dct(signal, config.transform), config.transform)[0]
        usable = (
            ((length - config.transform.patch_length) // config.transform.stride)
            * config.transform.stride
            + config.transform.patch_length
            if length >= config.transform.patch_length
            else config.transform.patch_length
        )
        central = recovered[
            config.transform.pad_length : config.transform.pad_length + usable
        ]
        error = float(np.max(np.abs(central - signal[:usable])))
        print(f"length={length} transform_max_error={error:.3e}")
        if error > 1e-4:
            raise RuntimeError("ST-DCT round-trip error is too large")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.resolve_paths(data_root=str(args.data_root) if args.data_root else None)
    print(f"data_root={config.data.root}")
    if config.data.kind == "wavelet_mat":
        check_wavelet(config)
    elif config.data.kind == "stead_hdf5":
        check_stead(config)
    else:
        raise ValueError(f"Unknown data kind: {config.data.kind}")
    check_transform(config)


if __name__ == "__main__":
    main()
