from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.transforms import normalize_std  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SeisDN HDF5 data")
    parser.add_argument("--seisdn-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat-file", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--val-mat-file", type=Path)
    parser.add_argument("--train-prefix", default="train")
    parser.add_argument("--val-prefix", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_tracebus(repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(
        "seisdn_tracebus", repo / "seisDN" / "preprocessing" / "TraceBusAA.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceBusAA


def trace_to_spectrum(tracebus, trace: np.ndarray):
    tensor = torch.from_numpy(trace[None, :].astype(np.float32))
    amplitudes, angles, _, _, _, _ = tracebus.trace2spectrogram(tensor, None)
    complex_spectrum = amplitudes * torch.exp(1j * angles)
    return complex_spectrum[0]


def build_split(
    tracebus,
    mat_file: Path,
    num_samples: int,
    snr_db: float | None,
    seed: int,
    output_path: Path,
    save_waveforms: bool,
) -> None:
    data = loadmat(str(mat_file))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    time_samples = 256
    spectra = np.zeros((num_samples, 6, 65, 64), dtype=np.float32)
    clean_waveforms = np.zeros((num_samples, time_samples), dtype=np.float32)
    noisy_waveforms = np.zeros_like(clean_waveforms)
    for index in range(num_samples):
        rng = np.random.default_rng(seed + 104729 * index)
        signal = signals[int(rng.integers(signals.shape[0]))]
        noise = noises[int(rng.integers(noises.shape[0]))]
        center = int(np.argmax(np.abs(signal)))
        start = int(np.clip(center - time_samples // 2, 0, signal.size - time_samples))
        clean = signal[start : start + time_samples]
        noise_start = int(rng.integers(0, noise.size - time_samples + 1))
        noise = noise[noise_start : noise_start + time_samples]
        snr = float(rng.uniform(-10.0, 10.0)) if snr_db is None else float(snr_db)
        noisy = clean + (10.0 ** (-snr / 20.0)) * noise
        clean_waveforms[index] = clean
        noisy_waveforms[index] = noisy
        clean_spectrum = trace_to_spectrum(tracebus, clean)
        noisy_spectrum = trace_to_spectrum(tracebus, noisy)
        if clean_spectrum.shape != noisy_spectrum.shape:
            raise RuntimeError("Clean/noisy spectra shape mismatch")
        if clean_spectrum.shape != (65, 64):
            raise RuntimeError(f"Unexpected spectrum shape: {clean_spectrum.shape}")
        spectra[index, 0] = noisy_spectrum.real.numpy()
        spectra[index, 1] = noisy_spectrum.imag.numpy()
        spectra[index, 4] = clean_spectrum.real.numpy()
        spectra[index, 5] = clean_spectrum.imag.numpy()
        if (index + 1) % 100 == 0:
            print(f"prepared {index + 1}/{num_samples}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("spectrograms", data=spectra)
    if save_waveforms:
        output_path.parent.joinpath("clean.npy").parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path.parent / "clean.npy", clean_waveforms)
        np.save(output_path.parent / "noisy.npy", noisy_waveforms)
    print(f"saved {output_path}", flush=True)


def build_cached_split(
    tracebus,
    data_dir: Path,
    prefix: str,
    num_samples: int,
    seed: int,
    output_path: Path,
    save_waveforms: bool,
) -> None:
    clean_cache = np.load(data_dir / f"{prefix}_clean.npy")
    noisy_cache = np.load(data_dir / f"{prefix}_noisy.npy")
    time_samples = clean_cache.shape[1]
    spectra = np.zeros((num_samples, 6, 65, 64), dtype=np.float32)
    clean_waveforms = np.zeros((num_samples, time_samples), dtype=np.float32)
    noisy_waveforms = np.zeros_like(clean_waveforms)
    for index in range(num_samples):
        rng = np.random.default_rng(seed + 104729 * index)
        section_index = int(rng.integers(clean_cache.shape[0]))
        trace = int(rng.integers(clean_cache.shape[2]))
        clean = clean_cache[section_index, :, trace].astype(np.float32)
        noisy = noisy_cache[section_index, :, trace].astype(np.float32)
        clean_waveforms[index] = clean
        noisy_waveforms[index] = noisy
        clean_spectrum = trace_to_spectrum(tracebus, clean)
        noisy_spectrum = trace_to_spectrum(tracebus, noisy)
        if clean_spectrum.shape != (65, 64):
            raise RuntimeError(f"Unexpected spectrum shape: {clean_spectrum.shape}")
        spectra[index, 0] = noisy_spectrum.real.numpy()
        spectra[index, 1] = noisy_spectrum.imag.numpy()
        spectra[index, 4] = clean_spectrum.real.numpy()
        spectra[index, 5] = clean_spectrum.imag.numpy()
        if (index + 1) % 100 == 0:
            print(f"prepared {index + 1}/{num_samples}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("spectrograms", data=spectra)
    if save_waveforms:
        np.save(output_path.parent / "clean.npy", clean_waveforms)
        np.save(output_path.parent / "noisy.npy", noisy_waveforms)
    print(f"saved {output_path}", flush=True)


def main() -> None:
    args = parse_args()
    TraceBusAA = load_tracebus(args.seisdn_root.resolve())
    tracebus = TraceBusAA(
        fs=125,
        freq_min=1,
        freq_max=60,
        target_fs=125,
        n_fft=128,
        hop_length=10,
        win_length=128,
        win_type="hann",
        pad_length=630,
    )
    protocol = {
        "train_source": str(args.mat_file)
        if args.mat_file is not None
        else str(args.data_dir),
        "validation_source": str(args.val_mat_file or args.mat_file),
        "num_train": args.num_train,
        "num_validation": args.num_val,
        "seed": args.seed,
        "trace_samples": 256,
        "stft": {
            "fs": 125,
            "freq_min": 1,
            "freq_max": 60,
            "target_fs": 125,
            "n_fft": 128,
            "hop_length": 10,
            "win_length": 128,
            "win_type": "hann",
            "pad_length": 630,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    if args.mat_file is not None:
        build_split(
            tracebus,
            args.mat_file,
            args.num_train,
            snr_db=None,
            seed=args.seed,
            output_path=args.output_dir / "train" / "train.h5",
            save_waveforms=False,
        )
        build_split(
            tracebus,
            args.val_mat_file or args.mat_file,
            args.num_val,
            snr_db=0.0,
            seed=args.seed + 10000,
            output_path=args.output_dir / "val" / "val.h5",
            save_waveforms=True,
        )
    else:
        build_cached_split(
            tracebus,
            args.data_dir,
            args.train_prefix,
            args.num_train,
            args.seed,
            args.output_dir / "train" / "train.h5",
            save_waveforms=False,
        )
        build_cached_split(
            tracebus,
            args.data_dir,
            args.val_prefix,
            args.num_val,
            args.seed + 10000,
            args.output_dir / "val" / "val.h5",
            save_waveforms=True,
        )


if __name__ == "__main__":
    main()
