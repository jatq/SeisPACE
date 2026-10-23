from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.calibrator1d import SeisPACECalibrator1D  # noqa: E402
from seispace.config import load_config  # noqa: E402
from seispace.engine import load_dual_sparse_prior  # noqa: E402
from seispace.prior2d import DualSparsePrior2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the serial SeisPACE path")
    parser.add_argument("mode", choices=["1d", "2d"])
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--scrn-root", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(callable_: object, warmup: int, repetitions: int, device: torch.device):
    for _ in range(warmup):
        callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repetitions):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        callable_()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - started)
    result = {
        "mean_ms": float(np.mean(samples) * 1000.0),
        "median_ms": float(np.median(samples) * 1000.0),
        "std_ms": float(np.std(samples) * 1000.0),
    }
    if device.type == "cuda":
        result["peak_memory_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )
    return result


def benchmark_1d(args: argparse.Namespace, device: torch.device) -> dict:
    config = load_config(args.prior_config)
    config.resolve_paths(
        output_dir=str(args.output.parent) if args.output.parent else None
    )
    prior = load_dual_sparse_prior(args.prior_checkpoint, config, device)
    refiner = SeisPACECalibrator1D(in_channels=2, residual=False, zero_head=False).to(
        device
    ).eval()
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    noisy = torch.randn(1, 2048, device=device)
    noisy_np = noisy.cpu().numpy()[0]

    def serial_path() -> None:
        with torch.no_grad():
            prior_output = np.asarray(prior.predict_signal(noisy_np))
            if prior_output.ndim == 1:
                prior_output = prior_output[None]
            prior_output = torch.from_numpy(prior_output).to(device)
            refiner(
                torch.cat(
                    [
                        noisy.unsqueeze(1),
                        prior_output[:, :2048].unsqueeze(1),
                    ],
                    dim=1,
                )
            )

    result = {
        "mode": "1d",
        "prior_parameters": parameter_count(prior),
        "refiner_parameters": parameter_count(refiner),
        "total_parameters": parameter_count(prior) + parameter_count(refiner),
        "latency": measure(serial_path, args.warmup, args.repetitions, device),
    }
    if args.baseline_checkpoint is not None:
        baseline = SeisPACECalibrator1D(
            in_channels=1, residual=False, zero_head=False
        ).to(device).eval()
        baseline.load_state_dict(
            torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        )
        result["baseline_parameters"] = parameter_count(baseline)

        def baseline_path() -> None:
            with torch.no_grad():
                baseline(noisy.unsqueeze(1))

        result["baseline_latency"] = measure(
            baseline_path,
            args.warmup,
            args.repetitions,
            device,
        )
    return result


def benchmark_2d(args: argparse.Namespace, device: torch.device) -> dict:
    if args.scrn_root is None:
        raise ValueError("--scrn-root is required for 2D benchmarking")
    config = json.loads(args.prior_config.read_text(encoding="utf-8"))["model"]
    prior = DualSparsePrior2D(
        patch_height=int(config["patch_height"]),
        patch_width=int(config["patch_width"]),
        stride_height=int(config["stride_height"]),
        stride_width=int(config["stride_width"]),
        embed_dim=int(config["embed_dim"]),
        dae_channels=tuple(config["dae_channels"]),
        init_embedding=bool(config["init_embedding"]),
        embed_loss_weight=float(config["embed_loss_weight"]),
    ).to(device).eval()
    prior.load_state_dict(
        torch.load(args.prior_checkpoint, map_location=device, weights_only=False)
    )
    scrn_module = load_module(
        "scrn_benchmark", args.scrn_root.resolve() / "model" / "SCRN.py"
    )
    refiner = scrn_module.SCRN(in_nc=2).to(device)
    refiner.m_tail[0] = nn.Conv2d(
        refiner.dim, 1, 3, 1, 1, bias=False
    ).to(device)
    refiner.load_state_dict(
        torch.load(args.refiner_checkpoint, map_location=device, weights_only=False)
    )
    refiner.eval()
    noisy = torch.randn(1, 1, 256, 128, device=device)

    def serial_path() -> None:
        with torch.no_grad():
            prior_output = prior.denoise(noisy)
            refiner(torch.cat([noisy, prior_output], dim=1))

    result = {
        "mode": "2d",
        "prior_parameters": parameter_count(prior),
        "refiner_parameters": parameter_count(refiner),
        "total_parameters": parameter_count(prior) + parameter_count(refiner),
        "latency": measure(serial_path, args.warmup, args.repetitions, device),
    }
    if args.baseline_checkpoint is not None:
        baseline = scrn_module.SCRN().to(device).eval()
        baseline.load_state_dict(
            torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        )
        result["baseline_parameters"] = parameter_count(baseline)

        def baseline_path() -> None:
            with torch.no_grad():
                baseline(noisy)

        result["baseline_latency"] = measure(
            baseline_path,
            args.warmup,
            args.repetitions,
            device,
        )
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = benchmark_1d(args, device) if args.mode == "1d" else benchmark_2d(
        args, device
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
