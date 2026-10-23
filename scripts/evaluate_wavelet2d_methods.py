from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.classical import (  # noqa: E402
    curvelet_denoise_2d,
    fx_decon_2d,
    wavelet_denoise_1d,
)
from seispace.config import load_config  # noqa: E402
from seispace.engine import load_dual_sparse_prior  # noqa: E402
from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrained 2D methods")
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--scrn-checkpoint", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-example", type=Path)
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(args.scrn_root.resolve()))
    scrn_module = load_module(
        "scrn_network_eval", args.scrn_root.resolve() / "model" / "SCRN.py"
    )
    scrn = scrn_module.SCRN().to(device)
    scrn.load_state_dict(
        torch.load(args.scrn_checkpoint, map_location=device, weights_only=False)
    )
    scrn.eval()

    prior_model = None
    if (args.prior_config is None) != (args.prior_checkpoint is None):
        raise ValueError(
            "--prior-config and --prior-checkpoint must be provided together"
        )
    if args.prior_config is not None:
        prior_model = load_dual_sparse_prior(
            args.prior_checkpoint,
            load_config(args.prior_config),
            device,
        )

    dataset = WaveletSectionDataset(
        args.test_mat,
        args.num_samples,
        training=False,
        snr_db=args.snr_db,
        seed=12345,
    )
    results = {
        "noisy": [],
        "scrn": [],
        "fx_decon": [],
        "curvelet": [],
        "wavelet": [],
    }
    if prior_model is not None:
        results["dual_sparse_prior"] = []
    saved_example = None
    for index in range(args.num_samples):
        clean, noisy = dataset[index]
        with torch.no_grad():
            scrn_out = scrn(
                torch.from_numpy(noisy)[None, None].to(device)
            ).cpu().numpy()[0, 0]
            if prior_model is not None:
                prior_traces = np.stack(
                    [
                        prior_model.predict_signal(noisy[:, trace])
                        for trace in range(noisy.shape[1])
                    ],
                    axis=1,
                )
        methods = {
            "noisy": noisy,
            "scrn": scrn_out,
            "fx_decon": fx_decon_2d(noisy),
            "curvelet": curvelet_denoise_2d(noisy),
            "wavelet": np.stack(
                [wavelet_denoise_1d(noisy[:, trace]) for trace in range(noisy.shape[1])],
                axis=1,
            ),
        }
        if prior_model is not None:
            methods["dual_sparse_prior"] = prior_traces
        for name, estimate in methods.items():
            scale = optimal_scale(clean, estimate)
            results[name].append({
                "snr_db": snr_db(clean, estimate),
                "ssim": structural_similarity(clean, estimate),
                "correlation": correlation(clean, estimate),
                "amplitude_ratio": float(np.std(estimate) / (np.std(clean) + 1e-12)),
                "optimal_scale": scale,
                "scaled_snr_db": snr_db(clean, scale * estimate),
            })
        if args.save_example is not None and saved_example is None:
            saved_example = {
                "clean": clean,
                "noisy": noisy,
                **{name: value for name, value in methods.items() if name != "noisy"},
            }
        if (index + 1) % 50 == 0:
            print(f"evaluated {index + 1}/{args.num_samples}", flush=True)

    summary = {
        name: {
            "snr_db": float(np.mean([item["snr_db"] for item in values])),
            "ssim": float(np.mean([item["ssim"] for item in values])),
            "correlation": float(np.mean([item["correlation"] for item in values])),
            "amplitude_ratio": float(
                np.mean([item["amplitude_ratio"] for item in values])
            ),
            "optimal_scale": float(
                np.mean([item["optimal_scale"] for item in values])
            ),
            "scaled_snr_db": float(
                np.mean([item["scaled_snr_db"] for item in values])
            ),
        }
        for name, values in results.items()
    }
    output = {
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "shape": [256, 128],
            "source": "wavelet-derived 2D sections",
            "ssim": "local Gaussian-window SSIM with reference data range",
        },
        "metrics": summary,
    }
    print(json.dumps(output, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.save_example is not None and saved_example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **saved_example)


if __name__ == "__main__":
    main()
