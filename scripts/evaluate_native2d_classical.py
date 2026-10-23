from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.classical import (  # noqa: E402
    curvelet_denoise_2d,
    fx_decon_2d,
    wavelet_denoise_1d,
)
from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate untrained classical methods on cached 2D sections"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["curvelet", "fx_decon", "wavelet"],
        default=["curvelet", "fx_decon", "wavelet"],
    )
    parser.add_argument("--curvelet-threshold", type=float, default=0.6)
    parser.add_argument("--fx-filter-length", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def metrics(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    scale = optimal_scale(clean, estimate)
    return {
        "snr_db": snr_db(clean, estimate),
        "ssim": structural_similarity(clean, estimate),
        "correlation": correlation(clean, estimate),
        "amplitude_ratio": float(
            np.std(estimate) / (np.std(clean) + 1e-12)
        ),
        "optimal_scale": scale,
        "scaled_snr_db": snr_db(clean, scale * estimate),
    }


def main() -> None:
    args = parse_args()
    clean = np.load(args.data_dir / f"{args.prefix}_clean.npy").astype(np.float32)
    noisy = np.load(args.data_dir / f"{args.prefix}_noisy.npy").astype(np.float32)
    count = min(
        len(clean),
        args.max_samples if args.max_samples is not None else len(clean),
    )
    rows: dict[str, list[dict[str, float]]] = {
        "noisy": [metrics(clean[index], noisy[index]) for index in range(count)]
    }
    for name in args.methods:
        values = []
        for index in range(count):
            if name == "curvelet":
                estimate = curvelet_denoise_2d(
                    noisy[index], threshold_factor=args.curvelet_threshold
                )
            elif name == "fx_decon":
                estimate = fx_decon_2d(
                    noisy[index], filter_length=args.fx_filter_length
                )
            else:
                estimate = np.stack(
                    [
                        wavelet_denoise_1d(noisy[index, :, trace])
                        for trace in range(noisy.shape[2])
                    ],
                    axis=1,
                )
            values.append(metrics(clean[index], estimate))
            if (index + 1) % 25 == 0:
                print(f"{name}: {index + 1}/{count}", flush=True)
        rows[name] = values

    summary = {
        "protocol": {
            "data_dir": str(args.data_dir),
            "prefix": args.prefix,
            "samples": count,
            "training": "none",
            "curvelet_threshold": args.curvelet_threshold,
            "fx_filter_length": args.fx_filter_length,
        },
        "metrics": {
            name: {
                key: float(np.mean([row[key] for row in values]))
                for key in values[0]
            }
            for name, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
