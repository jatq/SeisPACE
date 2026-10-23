from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.config import load_config  # noqa: E402
from seispace.engine import load_dual_sparse_prior  # noqa: E402
from seispace.transforms import normalize_std  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 1D SeisPACE training data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def generate_split(
    model,
    config,
    signals: np.ndarray,
    noises: np.ndarray,
    count: int,
    snr_db: float | None,
    seed: int,
    device: torch.device,
    output_dir: Path,
    prefix: str,
) -> None:
    rng = np.random.default_rng(seed)
    clean = np.empty((count, signals.shape[1]), dtype=np.float32)
    noisy = np.empty_like(clean)
    prior = np.empty_like(clean)
    for index in range(count):
        signal = signals[int(rng.integers(signals.shape[0]))]
        noise = noises[int(rng.integers(noises.shape[0]))]
        snr = float(rng.uniform(-10.0, 10.0)) if snr_db is None else float(snr_db)
        scale = 10.0 ** (-snr / 20.0)
        clean[index] = signal
        noisy[index] = signal + scale * noise
    with torch.no_grad():
        for start in range(0, count, 64):
            stop = min(count, start + 64)
            prior[start:stop] = np.stack(
                [model.predict_signal(trace) for trace in noisy[start:stop]],
                axis=0,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_clean.npy", clean)
    np.save(output_dir / f"{prefix}_noisy.npy", noisy)
    np.save(output_dir / f"{prefix}_prior.npy", prior)
    print(f"saved {prefix}: {count}", flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_dual_sparse_prior(args.checkpoint, config, device)
    data = loadmat(str(args.mat_file))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    generate_split(
        model,
        config,
        signals,
        noises,
        args.num_train,
        None,
        args.seed,
        device,
        args.output_dir,
        "train",
    )
    for snr in (-10.0, 0.0, 10.0):
        generate_split(
            model,
            config,
            signals,
            noises,
            args.num_val,
            snr,
            args.seed + 9000 + int(snr),
            device,
            args.output_dir,
            f"val_m{abs(int(snr))}" if snr < 0 else f"val_p{int(snr)}",
        )
    protocol = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "num_train": args.num_train,
        "num_val_each_snr": args.num_val,
        "train_snr_range": [-10.0, 10.0],
        "validation_snr_db": [-10.0, 0.0, 10.0],
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
