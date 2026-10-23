from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.metrics import correlation, optimal_scale, snr_db  # noqa: E402
from seispace.transforms import normalize_std  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrained SeisDN on held-out traces")
    parser.add_argument("--seisdn-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-mat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_tracebus(repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(
        "seisdn_tracebus_eval", repo / "seisDN" / "preprocessing" / "TraceBusAA.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceBusAA


def trace_to_spectrum(tracebus, trace: np.ndarray):
    tensor = torch.from_numpy(trace[None, :].astype(np.float32))
    amplitudes, angles, _, _, _, _ = tracebus.trace2spectrogram(tensor, None)
    return amplitudes[0], angles[0]


def recover_trace(tracebus, amplitudes, angles) -> np.ndarray:
    return tracebus.spectrogram2trace(
        amplitudes[None], angles[None], 0.0, 1.0
    )[0].numpy()


def normalize_trace(trace: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float32)
    trace = trace - trace.mean()
    return trace / max(float(np.max(np.abs(trace))), 1e-12)


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
    data = loadmat(str(args.test_mat))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    rows = []
    example = None
    for index in range(args.num_samples):
        rng = np.random.default_rng(args.seed + 104729 * index)
        signal = signals[int(rng.integers(signals.shape[0]))]
        noise = noises[int(rng.integers(noises.shape[0]))]
        center = int(np.argmax(np.abs(signal)))
        start = int(np.clip(center - 128, 0, signal.size - 256))
        clean = signal[start : start + 256]
        noise_start = int(rng.integers(0, noise.size - 256 + 1))
        noise = noise[noise_start : noise_start + 256]
        noisy = clean + (10.0 ** (-args.snr_db / 20.0)) * noise

        clean_spectrum, clean_angles = trace_to_spectrum(tracebus, clean)
        noisy_spectrum, noisy_angles = trace_to_spectrum(tracebus, noisy)
        with torch.no_grad():
            ratio = model(noisy_spectrum[1:][None, None].to(device))[0, 0].cpu()
        denoised_spectrum = noisy_spectrum.clone()
        denoised_spectrum[1:] = ratio * noisy_spectrum[1:]

        clean_roundtrip = normalize_trace(
            recover_trace(tracebus, clean_spectrum, clean_angles)
        )
        noisy_roundtrip = normalize_trace(
            recover_trace(tracebus, noisy_spectrum, noisy_angles)
        )
        denoised = normalize_trace(
            recover_trace(tracebus, denoised_spectrum, noisy_angles)
        )
        scale = optimal_scale(clean_roundtrip, denoised)
        rows.append(
            {
                "snr_noisy_db": snr_db(clean_roundtrip, noisy_roundtrip),
                "snr_denoised_db": snr_db(clean_roundtrip, denoised),
                "snr_gain_db": snr_db(clean_roundtrip, denoised)
                - snr_db(clean_roundtrip, noisy_roundtrip),
                "correlation": correlation(clean_roundtrip, denoised),
                "amplitude_ratio": float(
                    np.std(denoised) / (np.std(clean_roundtrip) + 1e-12)
                ),
                "scaled_snr_db": snr_db(clean_roundtrip, scale * denoised),
            }
        )
        if index == 0:
            example = {
                "clean": clean_roundtrip,
                "noisy": noisy_roundtrip,
                "seisdn": denoised,
            }
        if (index + 1) % 50 == 0:
            print(f"evaluated {index + 1}/{args.num_samples}", flush=True)

    summary = {
        "method": "SeisDN retrained from scratch, canonical STFT parameters",
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "seed": args.seed,
            "trace_length": 256,
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
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
