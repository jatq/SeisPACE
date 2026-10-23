from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.baselines.msmha import MSMHA  # noqa: E402
from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrained MSMHA section model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-mat", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--prefix", default="test_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def patch_inference(
    model: MSMHA,
    section: np.ndarray,
    patch_size: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    height, width = section.shape
    padded_height = (
        int(np.ceil((height - patch_size) / stride)) * stride + patch_size
    )
    padded_width = (
        int(np.ceil((width - patch_size) / stride)) * stride + patch_size
    )
    tensor = torch.from_numpy(section).float().to(device)
    tensor = F.pad(
        tensor[None, None],
        (0, padded_width - width, 0, padded_height - height),
        mode="reflect",
    )[0, 0]
    batch = F.unfold(
        tensor[None, None], kernel_size=patch_size, stride=stride
    )[0].T
    batch = batch.reshape(-1, patch_size, patch_size)
    outputs = []
    with torch.no_grad():
        for start in range(0, batch.shape[0], 256):
            _, prediction = model(batch[start : start + 256])
            outputs.append(prediction.cpu())
    output_patches = torch.cat(outputs)

    output = F.fold(
        output_patches.reshape(output_patches.shape[0], -1).T.unsqueeze(0),
        output_size=(padded_height, padded_width),
        kernel_size=patch_size,
        stride=stride,
    )
    count = F.fold(
        torch.ones(
            output_patches.shape[0], patch_size * patch_size
        ).T.unsqueeze(0),
        output_size=(padded_height, padded_width),
        kernel_size=patch_size,
        stride=stride,
    )
    return (output / count)[0, 0, :height, :width].cpu().numpy()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = MSMHA(
        patch_height=args.patch_size,
        patch_width=args.patch_size,
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=False)
    )
    model.eval()
    dataset = None
    if args.test_mat is not None:
        dataset = WaveletSectionDataset(
            args.test_mat,
            args.num_samples,
            training=False,
            snr_db=args.snr_db,
            seed=12345,
        )
    else:
        clean_cache = np.load(args.data_dir / f"{args.prefix}_clean.npy")
        noisy_cache = np.load(args.data_dir / f"{args.prefix}_noisy.npy")
        if args.num_samples > len(clean_cache):
            raise ValueError("num-samples exceeds cached section count")
    rows = []
    example = None
    for index in range(args.num_samples):
        if dataset is not None:
            clean, noisy = dataset[index]
        else:
            clean = clean_cache[index].astype(np.float32)
            noisy = noisy_cache[index].astype(np.float32)
        estimate = patch_inference(model, noisy, args.patch_size, args.stride, device)
        scale = optimal_scale(clean, estimate)
        rows.append(
            {
                "snr_noisy_db": snr_db(clean, noisy),
                "snr_db": snr_db(clean, estimate),
                "ssim": structural_similarity(clean, estimate),
                "correlation": correlation(clean, estimate),
                "amplitude_ratio": float(np.std(estimate) / (np.std(clean) + 1e-12)),
                "optimal_scale": scale,
                "scaled_snr_db": snr_db(clean, scale * estimate),
            }
        )
        if index == 0:
            example = {"clean": clean, "noisy": noisy, "msmha": estimate}
        if (index + 1) % 20 == 0:
            print(f"evaluated {index + 1}/{args.num_samples}", flush=True)
    summary = {
        "method": "MSMHA PyTorch reimplementation, trained from scratch",
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "data_dir": str(args.data_dir) if args.data_dir is not None else None,
            "prefix": args.prefix if args.data_dir is not None else None,
        },
        "metrics": {
            name: float(np.mean([row[name] for row in rows]))
            for name in rows[0]
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
