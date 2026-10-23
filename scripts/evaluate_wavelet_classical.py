from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.classical import wavelet_denoise_1d  # noqa: E402
from seispace.metrics import snr_gain  # noqa: E402
from seispace.transforms import mix_at_snr, normalize_std  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a classical wavelet baseline")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = loadmat(str(args.test_file))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    snrs = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    rows = []
    for snr in snrs:
        gains = []
        for index in range(signals.shape[0]):
            clean, noisy = mix_at_snr(
                signals[index], noises[index % noises.shape[0]], snr
            )
            denoised = wavelet_denoise_1d(noisy)
            gains.append(snr_gain(clean, noisy, denoised))
        rows.append((snr, float(np.mean(gains))))
        print(f"snr={snr:>3} dB mean_gain={rows[-1][1]:.2f} dB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *[str(snr) for snr, _ in rows]])
        writer.writerow(["wavelet-threshold", *[f"{gain:.2f}" for _, gain in rows]])


if __name__ == "__main__":
    main()
