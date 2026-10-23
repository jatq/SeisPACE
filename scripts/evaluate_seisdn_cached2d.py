from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SeisDN on entire cached 2D sections"
    )
    parser.add_argument("--seisdn-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_tracebus(repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(
        "seisdn_tracebus_cached2d",
        repo / "seisDN" / "preprocessing" / "TraceBusAA.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceBusAA


def normalize_traces(traces: torch.Tensor) -> torch.Tensor:
    traces = traces - traces.mean(dim=1, keepdim=True)
    return traces / torch.clamp(
        traces.abs().amax(dim=1, keepdim=True), min=1e-12
    )


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
    repo = args.seisdn_root.resolve()
    TraceBusAA = load_tracebus(repo)
    sys.path.insert(0, str(repo / "seisDN"))
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = config["model"]
    from models.model import Generator

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Generator(
        img_size=model_config["img_size"],
        patch_size=model_config["patch_size"],
        in_chans=model_config["in_chans"],
        embed_dim=model_config["embed_dim"],
        depths=model_config["depths"],
        num_heads=model_config["num_heads"],
        window_size=model_config["window_size"],
        mlp_ratio=model_config["mlp_ratio"],
        output_type=model_config["output_type"],
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    tracebus = TraceBusAA(
        fs=125,
        freq_min=1,
        freq_max=60,
        target_fs=125,
        n_fft=128,
        hop_length=10,
        win_length=128,
        win_type="hann",
        pad_length=630,
    )
    clean_cache = np.load(args.data_dir / f"{args.prefix}_clean.npy")
    noisy_cache = np.load(args.data_dir / f"{args.prefix}_noisy.npy")
    if args.num_samples > len(clean_cache):
        raise ValueError("num-samples exceeds cached section count")
    rows = []
    example = None
    for index in range(args.num_samples):
        clean = clean_cache[index].astype(np.float32)
        noisy = noisy_cache[index].astype(np.float32)
        clean_traces = torch.from_numpy(clean.T.copy())
        noisy_traces = torch.from_numpy(noisy.T.copy())
        clean_amp, clean_angle, _, _, _, _ = tracebus.trace2spectrogram(
            clean_traces, None
        )
        noisy_amp, noisy_angle, _, _, _, _ = tracebus.trace2spectrogram(
            noisy_traces, None
        )
        with torch.no_grad():
            ratio = model(noisy_amp[:, 1:].unsqueeze(1).to(device))[:, 0].cpu()
        denoised_amp = noisy_amp.clone()
        denoised_amp[:, 1:] = ratio * noisy_amp[:, 1:]
        clean_recovered = normalize_traces(
            tracebus.spectrogram2trace(clean_amp, clean_angle, 0.0, 1.0)
        )[:, : clean.shape[0]]
        noisy_recovered = normalize_traces(
            tracebus.spectrogram2trace(noisy_amp, noisy_angle, 0.0, 1.0)
        )[:, : clean.shape[0]]
        denoised_recovered = normalize_traces(
            tracebus.spectrogram2trace(denoised_amp, noisy_angle, 0.0, 1.0)
        )[:, : clean.shape[0]]
        clean_estimate = clean_recovered.T.numpy()
        noisy_estimate = noisy_recovered.T.numpy()
        denoised_estimate = denoised_recovered.T.numpy()
        rows.append(metrics(clean_estimate, denoised_estimate))
        if index == 0:
            example = {
                "clean": clean_estimate,
                "noisy": noisy_estimate,
                "seisdn": denoised_estimate,
            }
        if (index + 1) % 10 == 0:
            print(f"evaluated {index + 1}/{args.num_samples}", flush=True)

    summary = {
        "method": "SeisDN retrained on the same cached 2D source",
        "protocol": {
            "data_dir": str(args.data_dir),
            "prefix": args.prefix,
            "samples": args.num_samples,
            "inference": "all traces in each section, then section-level metrics",
            "stft": {
                "fs": 125,
                "freq_min": 1,
                "freq_max": 60,
                "n_fft": 128,
                "hop_length": 10,
                "win_length": 128,
                "pad_length": 630,
            },
        },
        "metrics": {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
