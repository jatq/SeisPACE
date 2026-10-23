from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)
from seispace.prior2d import DualSparsePrior2D  # noqa: E402
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 2D dual-sparse prior")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))["model"]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
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
    dataset = WaveletSectionDataset(
        args.test_mat,
        args.num_samples,
        training=False,
        snr_db=args.snr_db,
        seed=12345,
    )
    rows = []
    example = None
    with torch.no_grad():
        for index in range(args.num_samples):
            clean, noisy = dataset[index]
            estimate = model.denoise(
                torch.from_numpy(noisy)[None, None].to(device)
            )[0, 0].cpu().numpy()
            scale = optimal_scale(clean, estimate)
            rows.append(
                {
                    "snr_db": snr_db(clean, estimate),
                    "ssim": structural_similarity(clean, estimate),
                    "correlation": correlation(clean, estimate),
                    "amplitude_ratio": float(
                        np.std(estimate) / (np.std(clean) + 1e-12)
                    ),
                    "optimal_scale": scale,
                    "scaled_snr_db": snr_db(clean, scale * estimate),
                }
            )
            if index == 0:
                example = {
                    "clean": clean,
                    "noisy": noisy,
                    "prior2d": estimate,
                }
            if (index + 1) % 20 == 0:
                print(f"evaluated {index + 1}/{args.num_samples}", flush=True)
    result = {
        "method": "2D dual-sparse prior, trained from scratch",
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "shape": [256, 128],
        },
        "metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        },
        "median_metrics": {
            name: float(np.median([row[name] for row in rows])) for name in rows[0]
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["metrics", "median_metrics"]}, indent=2))
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
