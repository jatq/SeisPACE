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
from seispace.prior_control import apply_prior_control  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cached 2D refiner on native-section patches"
    )
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--input-mode", choices=["noisy_prior", "noisy"], default="noisy_prior"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--prior-mode",
        choices=["correct", "zero", "shuffled"],
        default="correct",
    )
    parser.add_argument("--prior-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clean = np.load(args.data_dir / f"{args.prefix}_clean.npy").astype(np.float32)
    noisy = np.load(args.data_dir / f"{args.prefix}_noisy.npy").astype(np.float32)
    prior = np.load(args.data_dir / f"{args.prefix}_prior.npy").astype(np.float32)
    controlled_prior = (
        apply_prior_control(prior, args.prior_mode, args.prior_seed)
        if args.input_mode == "noisy_prior"
        else prior
    )
    module = load_module(
        "scrn_cached_eval", args.scrn_root.resolve() / "model" / "SCRN.py"
    )
    in_channels = 2 if args.input_mode == "noisy_prior" else 1
    model = module.SCRN(in_nc=in_channels).to(device)
    model.m_tail[0] = nn.Conv2d(model.dim, 1, 3, 1, 1, bias=False).to(device)
    model.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    model.eval()
    estimates = []
    with torch.no_grad():
        for start in range(0, clean.shape[0], args.batch_size):
            stop = min(clean.shape[0], start + args.batch_size)
            noise = torch.from_numpy(noisy[start:stop])[:, None].to(device)
            if args.input_mode == "noisy_prior":
                prior_tensor = torch.from_numpy(
                    controlled_prior[start:stop]
                )[:, None].to(device)
                inputs = torch.cat([noise, prior_tensor], dim=1)
            else:
                inputs = noise
            estimates.append(model(inputs)[:, 0].cpu().numpy())
    estimate = np.concatenate(estimates)
    rows = [metrics(clean[index], estimate[index]) for index in range(len(clean))]
    prior_rows = [
        metrics(clean[index], controlled_prior[index])
        for index in range(len(clean))
    ]
    method = "SeisPACE-2D" if in_channels == 2 else "Noisy-only SCRN"
    if in_channels == 2 and args.prior_mode != "correct":
        method += f" ({args.prior_mode} prior control)"
    result = {
        "method": method,
        "protocol": {
            "data_dir": str(args.data_dir),
            "prefix": args.prefix,
            "samples": int(clean.shape[0]),
            "input_mode": args.input_mode,
            "prior_mode": (
                args.prior_mode if args.input_mode == "noisy_prior" else "unused"
            ),
            "prior_seed": args.prior_seed,
            "output_ensemble": False,
        },
        "metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        },
        "prior_metrics": {
            name: float(np.mean([row[name] for row in prior_rows]))
            for name in prior_rows[0]
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_example is not None:
        if not 0 <= args.example_index < clean.shape[0]:
            raise IndexError("example-index is outside the cached section set")
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_example,
            clean=clean[args.example_index],
            noisy=noisy[args.example_index],
            prior=controlled_prior[args.example_index],
            estimate=estimate[args.example_index],
        )
    print(json.dumps({"metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
