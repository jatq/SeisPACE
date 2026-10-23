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

from seispace.metrics import correlation, optimal_scale, snr_db  # noqa: E402
from seispace.prior_control import apply_prior_control  # noqa: E402
from seispace.calibrator1d import SeisPACECalibrator1D  # noqa: E402
from seispace.config import load_config  # noqa: E402
from seispace.engine import load_dual_sparse_prior  # noqa: E402
from seispace.transforms import normalize_std  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 1D SeisPACE calibrator")
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument(
        "--prior-mode",
        choices=["correct", "zero", "shuffled"],
        default="correct",
    )
    parser.add_argument("--prior-seed", type=int, default=2026)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--example-index", type=int, default=0)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--direct-refiner",
        action="store_true",
        help="Evaluate the main two-input warm-start/anchor refiner",
    )
    mode.add_argument(
        "--residual-refiner",
        action="store_true",
        help="Evaluate the exploratory zero-initialized residual refiner",
    )
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate the one-input direct U-Net baseline",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.prior_config)
    prior_model = load_dual_sparse_prior(args.prior_checkpoint, config, device)
    refiner = SeisPACECalibrator1D(
        in_channels=1 if args.baseline_only else 2,
        residual=args.residual_refiner,
        zero_head=args.residual_refiner,
    ).to(device)
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    refiner.eval()
    data = loadmat(str(args.test_mat))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    scale = 10.0 ** (-args.snr_db / 20.0)
    clean_all = signals
    noisy_all = signals + scale * noises
    with torch.no_grad():
        prior = np.stack(
            [prior_model.predict_signal(trace) for trace in noisy_all],
            axis=0,
        )
        if args.baseline_only:
            inputs = torch.from_numpy(noisy_all[:, None].astype(np.float32)).to(device)
        else:
            conditioned_prior = apply_prior_control(
                prior, args.prior_mode, args.prior_seed
            )
            inputs = torch.from_numpy(
                np.stack([noisy_all, conditioned_prior], axis=1).astype(np.float32)
            ).to(device)
        estimate = refiner(inputs)[0][:, 0].cpu().numpy()
    def collect(estimates: np.ndarray) -> dict[str, float]:
        rows = []
        for index in range(signals.shape[0]):
            scale_factor = optimal_scale(clean_all[index], estimates[index])
            rows.append(
                {
                    "snr_db": snr_db(clean_all[index], estimates[index]),
                    "gain_db": snr_db(clean_all[index], estimates[index])
                    - snr_db(clean_all[index], noisy_all[index]),
                    "correlation": correlation(clean_all[index], estimates[index]),
                    "amplitude_ratio": float(
                        np.std(estimates[index]) / (np.std(clean_all[index]) + 1e-12)
                    ),
                    "optimal_scale": scale_factor,
                    "scaled_snr_db": snr_db(
                        clean_all[index], scale_factor * estimates[index]
                    ),
                }
            )
        return {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        }

    seispace_metrics = collect(estimate)
    if args.baseline_only:
        method = "1D direct U-Net baseline"
    elif args.direct_refiner:
        method = "SeisPACE-1D"
    else:
        method = "SeisPACE-1D residual ablation"
    if not args.baseline_only and args.prior_mode != "correct":
        method += f" ({args.prior_mode} prior control)"
    result = {
        "method": method,
        "protocol": {
            "samples": signals.shape[0],
            "snr_db": args.snr_db,
            "prior_mode": "not_used" if args.baseline_only else args.prior_mode,
            "prior_seed": args.prior_seed,
            "output_ensemble": False,
            "baseline_anchor_used_at_inference": False,
        },
        "metrics": seispace_metrics,
        "prior_metrics": collect(prior),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_example is not None:
        if not 0 <= args.example_index < signals.shape[0]:
            raise IndexError("example-index is outside the test set")
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_example,
            clean=clean_all[args.example_index],
            noisy=noisy_all[args.example_index],
            prior=prior[args.example_index],
            estimate=estimate[args.example_index],
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
