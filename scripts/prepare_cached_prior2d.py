from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.prior2d import DualSparsePrior2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace cached 2D priors using a retrained prior model"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=["train", "val", "test_m10", "test_0", "test_p10"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--preserve-amplitude",
        action="store_true",
        help="Keep the network output scale instead of per-section standardization",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = json.loads(args.config.read_text(encoding="utf-8"))["model"]
    model = DualSparsePrior2D(
        patch_height=int(config["patch_height"]),
        patch_width=int(config["patch_width"]),
        stride_height=int(config["stride_height"]),
        stride_width=int(config["stride_width"]),
        embed_dim=int(config["embed_dim"]),
        dae_channels=tuple(config["dae_channels"]),
        init_embedding=bool(config["init_embedding"]),
        embed_loss_weight=float(config["embed_loss_weight"]),
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    model.eval()
    for prefix in args.prefixes:
        noisy_path = args.data_dir / f"{prefix}_noisy.npy"
        prior_path = args.data_dir / f"{prefix}_prior.npy"
        synthetic_path = args.data_dir / f"{prefix}_prior_synthetic.npy"
        if prior_path.exists() and synthetic_path.exists():
            raise FileExistsError(
                f"Both field and synthetic prior files exist for {prefix}"
            )
        if prior_path.exists():
            prior_path.replace(synthetic_path)
        noisy = np.load(noisy_path)
        prior = np.empty_like(noisy)
        with torch.no_grad():
            for start in range(0, noisy.shape[0], args.batch_size):
                stop = min(noisy.shape[0], start + args.batch_size)
                batch = torch.from_numpy(
                    noisy[start:stop].astype(np.float32)
                )[:, None].to(device)
                result = model.denoise(batch).cpu().numpy()[:, 0]
                if not args.preserve_amplitude:
                    mean = result.mean(axis=(-2, -1), keepdims=True)
                    std = result.std(axis=(-2, -1), keepdims=True).clip(min=1e-8)
                    result = (result - mean) / std
                prior[start:stop] = result.astype(np.float16)
                print(f"{prefix}: {stop}/{noisy.shape[0]}", flush=True)
        np.save(prior_path, prior)


if __name__ == "__main__":
    main()
