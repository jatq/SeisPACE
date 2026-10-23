from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.metrics import (  # noqa: E402
    correlation,
    optimal_scale,
    snr_db,
    structural_similarity,
)
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional PS-SGMs denoising")
    parser.add_argument("--ps-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-mat", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--prefix", default="test_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-example", type=Path)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--step-lr", type=float)
    parser.add_argument("--n-steps-each", type=int)
    parser.add_argument("--clip-range", type=float, nargs=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def namespace(mapping: dict) -> Namespace:
    result = Namespace()
    for key, value in mapping.items():
        setattr(result, key, namespace(value) if isinstance(value, dict) else value)
    return result


@torch.no_grad()
def conditional_denoise(
    score,
    noisy_patches: torch.Tensor,
    sigmas: torch.Tensor,
    sigma_0: float,
    step_lr: float,
    n_steps_each: int,
    clip_range: tuple[float, float] | None,
) -> torch.Tensor:
    y_0 = noisy_patches.flatten(1)
    sigma_0 = float(np.clip(sigma_0, float(sigmas[-1]), float(sigmas[0])))
    start = int(torch.argmin(torch.abs(sigmas - sigma_0)).item())
    current = y_0.clone()
    for class_index in range(start, len(sigmas)):
        sigma = sigmas[class_index]
        labels = torch.full(
            (current.shape[0],), class_index, dtype=torch.long, device=current.device
        )
        denominator = sigma_0**2 - sigma**2
        minimum = float(sigmas[-1]) ** 2
        denominator = torch.where(
            torch.abs(denominator) < minimum,
            torch.full_like(denominator, minimum),
            denominator,
        )
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for _ in range(n_steps_each):
            score_value = score(
                current.view(-1, 1, *noisy_patches.shape[-2:]), labels
            ).flatten(1)
            gradient = score_value + (y_0 - current) / denominator
            current = current + step_size * gradient
            current = current + torch.randn_like(current) * torch.sqrt(2.0 * step_size)
            if clip_range is not None:
                current = torch.clamp(current, clip_range[0], clip_range[1])
    last_labels = torch.full(
        (current.shape[0],), len(sigmas) - 1, dtype=torch.long, device=current.device
    )
    current = current + sigmas[-1] ** 2 * score(
        current.view(-1, 1, *noisy_patches.shape[-2:]), last_labels
    ).flatten(1)
    return current.view_as(noisy_patches)


def extract_patches(section: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    tensor = torch.from_numpy(section[None, None]).float()
    patches = F.unfold(tensor, kernel_size=patch_size, stride=stride)[0].T
    return patches.reshape(-1, patch_size, patch_size).numpy()


def merge_patches(
    patches: np.ndarray,
    shape: tuple[int, int],
    patch_size: int,
    stride: int,
) -> np.ndarray:
    height, width = shape
    tensor = torch.from_numpy(patches[:, None]).float()
    output = F.fold(
        tensor.reshape(len(patches), -1).T.unsqueeze(0),
        output_size=(height, width),
        kernel_size=patch_size,
        stride=stride,
    )
    count = F.fold(
        torch.ones(len(patches), patch_size * patch_size).T.unsqueeze(0),
        output_size=(height, width),
        kernel_size=patch_size,
        stride=stride,
    )
    return (output / count)[0, 0].numpy()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = namespace(raw)
    config.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config.data.channels = 1
    config.data.image_size = config.data.patch_size
    config.data.image_shape = (config.data.patch_size, config.data.patch_size)
    config.data.logit_transform = False
    config.data.rescaled = False
    config.data.seis_rescaled = True

    ps_root = args.ps_root.resolve()
    sys.path.insert(0, str(ps_root))
    from models import get_sigmas
    from models.ema import EMAHelper
    from models.ncsnv2 import NCSNv2

    score = NCSNv2(config).to(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    score.load_state_dict(checkpoint["model"])
    ema = EMAHelper(mu=float(config.model.ema_rate))
    ema.register(score)
    ema.load_state_dict(checkpoint["ema"])
    ema.ema(score)
    score.eval()
    sigmas = get_sigmas(config)

    dataset = None
    if args.test_mat is not None:
        dataset = WaveletSectionDataset(
            args.test_mat,
            args.num_samples,
            training=False,
            snr_db=args.snr_db,
            seed=args.seed,
        )
    else:
        clean_cache = np.load(args.data_dir / f"{args.prefix}_clean.npy")
        noisy_cache = np.load(args.data_dir / f"{args.prefix}_noisy.npy")
        if args.num_samples > len(clean_cache):
            raise ValueError("num-samples exceeds cached section count")
    patch_size = int(config.data.patch_size)
    stride = patch_size // 2
    rows = []
    example = None
    for index in range(args.num_samples):
        if dataset is not None:
            clean, noisy = dataset[index]
        else:
            clean = clean_cache[index].astype(np.float32)
            noisy = noisy_cache[index].astype(np.float32)
        scale = max(float(np.max(np.abs(clean))), 1e-8)
        clean_normalized = clean / scale
        noisy_normalized = noisy / scale
        noisy_patches = extract_patches(noisy_normalized, patch_size, stride)
        transformed = torch.from_numpy(noisy_patches[:, None]).to(config.device)
        transformed = (transformed + 1.0) * 0.5
        sigma_0 = float(np.std((noisy_normalized - clean_normalized) * 0.5))
        if index == 0:
            print(f"first-section noise standard deviation in model domain={sigma_0:.4f}")
        denoised_patches = conditional_denoise(
            score,
            transformed,
            sigmas,
            sigma_0,
            args.step_lr
            if args.step_lr is not None
            else float(config.sampling.step_lr),
            args.n_steps_each
            if args.n_steps_each is not None
            else int(config.sampling.n_steps_each),
            tuple(args.clip_range) if args.clip_range is not None else None,
        )
        denoised_patches = denoised_patches.cpu().numpy()[:, 0] * 2.0 - 1.0
        # All patches share the same overlap pattern except the time edges.
        patch_rows = []
        cursor = 0
        for row in range(0, noisy_normalized.shape[0] - patch_size + 1, stride):
            for column in range(0, noisy_normalized.shape[1] - patch_size + 1, stride):
                patch_rows.append((row, column, denoised_patches[cursor]))
                cursor += 1
        accum = np.zeros_like(noisy_normalized)
        count = np.zeros_like(noisy_normalized)
        for row, column, patch in patch_rows:
            accum[row : row + patch_size, column : column + patch_size] += patch
            count[row : row + patch_size, column : column + patch_size] += 1.0
        denoised_normalized = accum / np.maximum(count, 1e-8)
        denoised = denoised_normalized * scale

        estimate_scale = optimal_scale(clean, denoised)
        rows.append(
            {
                "snr_db": snr_db(clean, denoised),
                "ssim": structural_similarity(clean, denoised),
                "correlation": correlation(clean, denoised),
                "amplitude_ratio": float(np.std(denoised) / (np.std(clean) + 1e-12)),
                "optimal_scale": estimate_scale,
                "scaled_snr_db": snr_db(clean, estimate_scale * denoised),
            }
        )
        if index == 0:
            example = {"clean": clean, "noisy": noisy, "ps_sgms": denoised}
        print(f"evaluated {index + 1}/{args.num_samples}", flush=True)

    result = {
        "method": "PS-SGMs score model retrained from scratch on our clean sections",
        "protocol": {
            "samples": args.num_samples,
            "snr_db": args.snr_db,
            "patch_size": patch_size,
            "patch_stride": stride,
            "checkpoint_iteration": int(checkpoint["iteration"]),
            "training_iterations": int(config.training.iterations),
            "variations": 1,
            "data_dir": str(args.data_dir) if args.data_dir is not None else None,
            "prefix": args.prefix if args.data_dir is not None else None,
        },
        "metrics": {
            name: float(np.mean([row[name] for row in rows])) for name in rows[0]
        },
        "median_metrics": {
            name: float(np.median([row[name] for row in rows])) for name in rows[0]
        },
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.save_example is not None and example is not None:
        args.save_example.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save_example, **example)


if __name__ == "__main__":
    main()
