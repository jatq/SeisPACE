from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the official SCRN checkpoint")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(
        10.0
        * np.log10(
            np.sum(reference**2) / np.sum((reference - estimate) ** 2)
        )
    )


def global_ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = reference.mean()
    mu_y = estimate.mean()
    var_x = reference.var()
    var_y = estimate.var()
    covariance = np.mean((reference - mu_x) * (estimate - mu_y))
    return float(
        ((2 * mu_x * mu_y + c1) * (2 * covariance + c2))
        / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    )


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = torch.load(repo / "trained_model" / "model.pth", map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    clear = np.load(repo / "test_data" / "clear.npy").astype(np.float32)
    noisy = np.load(repo / "test_data" / "noise_and_miss.npy").astype(np.float32)
    tensor = torch.from_numpy(noisy)[None, None].to(device)
    with torch.no_grad():
        denoised = model(tensor).detach().cpu().numpy()[0, 0]
    result = {
        "method": "SCRN",
        "snr_noisy_db": snr(clear, noisy),
        "snr_denoised_db": snr(clear, denoised),
        "ssim_noisy": global_ssim(clear, noisy),
        "ssim_denoised": global_ssim(clear, denoised),
        "shape": list(clear.shape),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
