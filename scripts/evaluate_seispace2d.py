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
from seispace.prior_control import apply_prior_control  # noqa: E402
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the SeisPACE 2D calibrator")
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument(
        "--input-mode",
        choices=["noisy_prior", "noisy"],
        default="noisy_prior",
    )
    parser.add_argument(
        "--prior-mode",
        choices=["correct", "zero", "shuffled"],
        default="correct",
    )
    parser.add_argument("--prior-seed", type=int, default=2026)
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
    scrn_module = load_module(
        "scrn_seispace_eval", args.scrn_root.resolve() / "model" / "SCRN.py"
    )
    in_channels = 2 if args.input_mode == "noisy_prior" else 1
    refiner = scrn_module.SCRN(in_nc=in_channels).to(device)
    refiner.m_tail[0] = nn.Conv2d(
        refiner.dim, 1, 3, 1, 1, bias=False
    ).to(device)
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    refiner.eval()

    dataset = WaveletSectionDataset(
        args.test_mat,
        args.num_samples,
        training=False,
        snr_db=args.snr_db,
        seed=12345,
    )
    clean_all = []
    noisy_all = []
    prior_all = []
    with torch.no_grad():
        for index in range(args.num_samples):
            clean, noisy = dataset[index]
            clean_all.append(clean)
            noisy_all.append(noisy)
            if args.input_mode == "noisy_prior":
                noisy_tensor = torch.from_numpy(noisy)[None, None].to(device)
                prior = dp.denoise(noisy_tensor)[0, 0].cpu().numpy()
                prior_all.append(prior)
            if (index + 1) % 20 == 0:
                print(f"prepared {index + 1}/{args.num_samples}", flush=True)

    clean_stack = np.stack(clean_all)
    noisy_stack = np.stack(noisy_all)
    prior_stack = np.stack(prior_all) if prior_all else None
    conditioned_prior = (
        apply_prior_control(prior_stack, args.prior_mode, args.prior_seed)
        if prior_stack is not None
        else None
    )

    rows = []
    prior_rows = []
    example = None
    with torch.no_grad():
        for index in range(args.num_samples):
            clean = clean_stack[index]
            noisy = noisy_stack[index]
            noisy_tensor = torch.from_numpy(noisy)[None, None].to(device)
            if args.input_mode == "noisy_prior":
                inputs = torch.cat(
                    [
                        noisy_tensor,
                        torch.from_numpy(conditioned_prior[index])[None, None].to(
                            device
                        ),
                    ],
                    dim=1,
                )
            else:
                inputs = noisy_tensor
            estimate = refiner(inputs)[0, 0].cpu().numpy()
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
            if prior_stack is not None:
                prior_estimate = prior_stack[index]
                prior_scale = optimal_scale(clean, prior_estimate)
                prior_rows.append(
                    {
                        "snr_db": snr_db(clean, prior_estimate),
                        "ssim": structural_similarity(clean, prior_estimate),
                        "correlation": correlation(clean, prior_estimate),
                        "amplitude_ratio": float(
                            np.std(prior_estimate) / (np.std(clean) + 1e-12)
                        ),
                        "optimal_scale": prior_scale,
                        "scaled_snr_db": snr_db(
                            clean, prior_scale * prior_estimate
                        ),
                    }
                )
            if index == 0:
                example = {
                    "clean": clean,
                    "noisy": noisy_stack[index],
                    "calibrate": estimate,
                }
                if prior_stack is not None:
                    example["prior2d"] = prior_stack[index]
            if (index + 1) % 20 == 0:
                print(f"evaluated {index + 1}/{args.num_samples}", flush=True)
    method = (
        "SeisPACE-2D"
        if args.input_mode == "noisy_prior"
        else "Noisy-only SCRN continuation"
    )
    if args.input_mode == "noisy_prior" and args.prior_mode != "correct":
        method += f" ({args.prior_mode} prior control)"
    result = {
        "method": method,
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "input_mode": args.input_mode,
            "prior_mode": (
                args.prior_mode
                if args.input_mode == "noisy_prior"
                else "not_used"
            ),
            "prior_seed": args.prior_seed,
            "output_ensemble": False,
            "baseline_anchor_used_at_inference": False,
        },
        "metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        },
        "median_metrics": {
            name: float(np.median([row[name] for row in rows])) for name in rows[0]
        },
        "samples": rows,
    }
    if prior_rows:
        result["prior_metrics"] = {
            name: float(np.mean([row[name] for row in prior_rows]))
            for name in prior_rows[0]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["metrics", "median_metrics"]}, indent=2))
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
