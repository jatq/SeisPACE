from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from seispace.data import read_csv, write_csv
from seispace.models import build_calibrator, build_prior


OUTPUT_COLUMNS = [
    "sample_id",
    "dimension",
    "shape",
    "clean",
    "noisy",
    "prior",
    "estimate",
    "snr_db",
    "gain_db",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SeisPACE inference with a bundled checkpoint."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/seispace.pt"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    signal_power = float(np.var(reference))
    noise_power = float(np.var(reference - estimate))
    return float(10.0 * np.log10((signal_power + 1e-12) / (noise_power + 1e-12)))


def load_component(
    bundle: dict,
    dimension: str,
    name: str,
    in_channels: int | None,
    device: torch.device,
) -> torch.nn.Module:
    component = bundle[dimension][name]
    config = component["config"]
    if name == "prior":
        model = build_prior(dimension, config)
    else:
        if in_channels is None:
            raise ValueError("in_channels is required for a calibrator")
        model = build_calibrator(dimension, in_channels, config)
    model.load_state_dict(component["state_dict"])
    return model.to(device).eval()


def predict(
    noisy: np.ndarray,
    dimension: str,
    prior_model: torch.nn.Module,
    calibrator: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(noisy))
    scale = max(float(np.std(noisy)), 1e-8)
    normalized = (noisy - mean) / scale
    tensor = torch.from_numpy(normalized[None, None].astype(np.float32)).to(device)
    with torch.no_grad():
        if dimension == "1d":
            prior = prior_model.predict_signal(normalized)
            prior_tensor = torch.from_numpy(prior[None, None]).to(device)
        else:
            prior_tensor = prior_model.denoise(tensor)
            prior = prior_tensor[0, 0].cpu().numpy()
        output = calibrator(torch.cat([tensor, prior_tensor], dim=1))
        estimate = output[0] if isinstance(output, tuple) else output
    estimate = estimate[0, 0].cpu().numpy() * scale + mean
    prior = np.asarray(prior, dtype=np.float32) * scale + mean
    return prior.astype(np.float32), estimate.astype(np.float32)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    samples = read_csv(args.input)
    dimension = samples[0].dimension
    bundle = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if bundle.get("format") != "seispace-bundle-v1":
        raise ValueError(f"{args.checkpoint} is not a SeisPACE release bundle")
    if dimension not in bundle:
        raise ValueError(f"Checkpoint does not contain {dimension} weights")

    prior_model = load_component(bundle, dimension, "prior", None, device)
    calibrator = load_component(bundle, dimension, "calibrator", 2, device)
    rows = []
    for sample in samples:
        prior, estimate = predict(
            sample.noisy,
            dimension,
            prior_model,
            calibrator,
            device,
        )
        row = {
            "sample_id": sample.sample_id,
            "dimension": dimension,
            "shape": sample.shape[0]
            if dimension == "1d"
            else f"{sample.shape[0]}x{sample.shape[1]}",
            "clean": sample.clean if sample.clean is not None else "",
            "noisy": sample.noisy,
            "prior": prior,
            "estimate": estimate,
            "snr_db": "",
            "gain_db": "",
        }
        if sample.clean is not None:
            row["snr_db"] = f"{snr_db(sample.clean, estimate):.6f}"
            row["gain_db"] = f"{snr_db(sample.clean, estimate) - snr_db(sample.clean, sample.noisy):.6f}"
        rows.append(row)
    write_csv(args.output, rows, OUTPUT_COLUMNS)
    print(f"saved {args.output} ({len(rows)} samples)", flush=True)


if __name__ == "__main__":
    main()
