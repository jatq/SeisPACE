from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    parser = argparse.ArgumentParser(description="Blend SCRN and SeisPACE predictions")
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--scrn-checkpoint", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scrn_module = load_module(
        "scrn_blend_eval", args.scrn_root.resolve() / "model" / "SCRN.py"
    )
    scrn = scrn_module.SCRN().to(device)
    scrn.load_state_dict(
        torch.load(args.scrn_checkpoint, map_location=device, weights_only=False)
    )
    scrn.eval()
    refiner = scrn_module.SCRN(in_nc=2).to(device)
    refiner.m_tail[0] = nn.Conv2d(
        refiner.dim, 1, 3, 1, 1, bias=False
    ).to(device)
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    refiner.eval()
    prior_config = json.loads(args.prior_config.read_text(encoding="utf-8"))["model"]
    dp = DualSparsePrior2D(
        patch_height=int(prior_config["patch_height"]),
        patch_width=int(prior_config["patch_width"]),
        stride_height=int(prior_config["stride_height"]),
        stride_width=int(prior_config["stride_width"]),
        embed_dim=int(prior_config["embed_dim"]),
        dae_channels=tuple(prior_config["dae_channels"]),
        init_embedding=bool(prior_config["init_embedding"]),
        embed_loss_weight=float(prior_config["embed_loss_weight"]),
    ).to(device)
    dp.load_state_dict(
        torch.load(args.prior_checkpoint, map_location=device, weights_only=False)
    )
    dp.eval()

    dataset = WaveletSectionDataset(
        args.test_mat,
        args.num_samples,
        training=False,
        snr_db=args.snr_db,
        seed=args.seed,
    )
    rows = {alpha: [] for alpha in args.alphas}
    example = None
    with torch.no_grad():
        for index in range(args.num_samples):
            clean, noisy = dataset[index]
            noisy_tensor = torch.from_numpy(noisy)[None, None].to(device)
            prior = dp.denoise(noisy_tensor)
            seispace = refiner(torch.cat([noisy_tensor, prior], dim=1))[0, 0].cpu().numpy()
            baseline = scrn(noisy_tensor)[0, 0].cpu().numpy()
            for alpha in args.alphas:
                estimate = alpha * baseline + (1.0 - alpha) * seispace
                scale = optimal_scale(clean, estimate)
                rows[alpha].append(
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
                if index == 0 and example is None:
                    example = {
                        "clean": clean,
                        "noisy": noisy,
                        "prior2d": prior[0, 0].cpu().numpy(),
                        "scrn": baseline,
                        f"blend_{alpha:.2f}": estimate,
                    }
            if (index + 1) % 20 == 0:
                print(f"evaluated {index + 1}/{args.num_samples}", flush=True)
    result = {
        "protocol": {"samples": args.num_samples, "snr_db": args.snr_db},
        "alphas": {
            str(alpha): {
                name: float(np.mean([row[name] for row in values]))
                for name in values[0]
            }
            for alpha, values in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
