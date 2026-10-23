from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.metrics import correlation, snr_db, structural_similarity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cached 2D prior")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(clean: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    rows = [
        {
            "snr_db": snr_db(clean[index], estimate[index]),
            "ssim": structural_similarity(clean[index], estimate[index]),
            "correlation": correlation(clean[index], estimate[index]),
            "amplitude_ratio": float(
                np.std(estimate[index]) / (np.std(clean[index]) + 1e-12)
            ),
        }
        for index in range(len(clean))
    ]
    return {
        name: float(np.mean([row[name] for row in rows])) for name in rows[0]
    }


def main() -> None:
    args = parse_args()
    clean = np.load(args.data_dir / f"{args.prefix}_clean.npy").astype(np.float32)
    noisy = np.load(args.data_dir / f"{args.prefix}_noisy.npy").astype(np.float32)
    prior = np.load(args.data_dir / f"{args.prefix}_prior.npy").astype(np.float32)
    result = {
        "prefix": args.prefix,
        "samples": int(len(clean)),
        "noisy": summarize(clean, noisy),
        "prior": summarize(clean, prior),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
