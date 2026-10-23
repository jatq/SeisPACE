from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import segyio
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.prior2d import DualSparsePrior2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a controlled-noise benchmark from a native SEGY section"
    )
    parser.add_argument("--segy", type=Path, required=True)
    parser.add_argument("--noise-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=200)
    parser.add_argument("--num-test", type=int, default=200)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--split-axis",
        choices=("time", "trace"),
        default="time",
        help="Axis used for train/validation/test separation",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_segy(path: Path) -> np.ndarray:
    with segyio.open(str(path), "r", ignore_geometry=True) as handle:
        traces = np.asarray(segyio.tools.collect(handle.trace[:]), dtype=np.float32)
    return traces.T


def load_noise_bank(path: Path, limit: int = 2048) -> list[np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle["data"]
        names = list(group.keys())[:limit]
        return [np.asarray(group[name], dtype=np.float32)[0] for name in names]


def standardize_block(block: np.ndarray) -> np.ndarray:
    centered = block - np.mean(block)
    scale = float(np.std(centered))
    if scale < 1e-6:
        raise ValueError("degenerate SEGY block")
    return (centered / scale).astype(np.float32)


def robust_normalize(block: np.ndarray) -> np.ndarray:
    low, high = np.percentile(block, [0.5, 99.5])
    clipped = np.clip(block, low, high)
    return standardize_block(clipped)


def random_clean_block(
    data: np.ndarray,
    time_start: int,
    time_stop: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    split_axis: str,
) -> np.ndarray:
    for _ in range(100):
        if split_axis == "time":
            trace = int(rng.integers(0, data.shape[1] - width + 1))
            start = int(rng.integers(time_start, time_stop - height + 1))
        else:
            trace = int(rng.integers(time_start, time_stop - width + 1))
            start = int(rng.integers(0, data.shape[0] - height + 1))
        block = data[start : start + height, trace : trace + width]
        try:
            return robust_normalize(block)
        except ValueError:
            continue
    raise RuntimeError("Unable to sample a non-degenerate SEGY block")


def noise_block(
    bank: list[np.ndarray],
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty((height, width), dtype=np.float32)
    for column in range(width):
        trace = bank[int(rng.integers(len(bank)))]
        start = int(rng.integers(0, trace.size - height + 1))
        output[:, column] = trace[start : start + height]
    return standardize_block(output)


def build_model(
    config_path: Path, checkpoint: Path, device: torch.device
) -> DualSparsePrior2D:
    config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    model = DualSparsePrior2D(
        patch_height=int(config["patch_height"]),
        patch_width=int(config["patch_width"]),
        stride_height=int(config["stride_height"]),
        stride_width=int(config["stride_width"]),
        embed_dim=int(config["embed_dim"]),
        dae_channels=tuple(config["dae_channels"]),
        init_embedding=bool(config["init_embedding"]),
        embed_loss_weight=float(config["embed_loss_weight"]),
    ).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)
    )
    return model.eval()


def generate_split(
    model: DualSparsePrior2D,
    data: np.ndarray,
    noise_bank: list[np.ndarray],
    output_dir: Path,
    prefix: str,
    count: int,
    time_range: tuple[int, int],
    height: int,
    width: int,
    seed: int,
    device: torch.device,
    batch_size: int,
    fixed_snr: float | None,
    split_axis: str,
) -> dict[str, float]:
    clean_all = np.empty((count, height, width), dtype=np.float16)
    noisy_all = np.empty_like(clean_all)
    prior_all = np.empty_like(clean_all)
    noise_scales = []
    for index in range(count):
        rng = np.random.default_rng(seed + 104729 * index)
        clean = random_clean_block(
            data,
            time_range[0],
            time_range[1],
            height,
            width,
            rng,
            split_axis,
        )
        snr = fixed_snr if fixed_snr is not None else float(rng.uniform(-10, 10))
        scale = 10.0 ** (-snr / 20.0)
        noisy = clean + scale * noise_block(noise_bank, height, width, rng)
        clean_all[index] = clean.astype(np.float16)
        noisy_all[index] = noisy.astype(np.float16)
        noise_scales.append(scale)

    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            noisy = torch.from_numpy(noisy_all[start:stop].astype(np.float32))[
                :, None
            ].to(device)
            prior = model.denoise(noisy).cpu().numpy()[:, 0]
            prior_all[start:stop] = prior.astype(np.float16)
            print(f"{prefix}: prepared {stop}/{count}", flush=True)

    np.save(output_dir / f"{prefix}_clean.npy", clean_all)
    np.save(output_dir / f"{prefix}_noisy.npy", noisy_all)
    np.save(output_dir / f"{prefix}_prior.npy", prior_all)
    return {
        "samples": count,
        "noise_scale_mean": float(np.mean(noise_scales)),
        "noise_scale_min": float(np.min(noise_scales)),
        "noise_scale_max": float(np.max(noise_scales)),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = load_segy(args.segy)
    if data.shape[0] < args.height or data.shape[1] < args.width:
        raise ValueError(f"SEGY section is too small: {data.shape}")
    noise_bank = load_noise_bank(args.noise_hdf5)
    model = build_model(args.prior_config, args.prior_checkpoint, device)
    split_size = (
        data.shape[0] if args.split_axis == "time" else data.shape[1]
    )
    block_size = args.height if args.split_axis == "time" else args.width
    if split_size < block_size:
        raise ValueError(
            f"Cannot split {args.split_axis} axis of size {split_size} "
            f"into blocks of size {block_size}"
        )
    train_stop = int(split_size * 0.60)
    val_stop = int(split_size * 0.70)
    ranges = {
        "train": (0, train_stop),
        "val": (train_stop, val_stop),
        "test": (val_stop, split_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "segy": str(args.segy),
        "native_shape_samples_traces": list(data.shape),
        "noise": str(args.noise_hdf5),
        "noise_domain": "STEAD real ambient-noise traces",
        "reference": "native SEGY section; controlled noise added for scoring",
        "split_axis": args.split_axis,
        "split_ranges": ranges,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "prior_config": str(args.prior_config),
        "prior_checkpoint": str(args.prior_checkpoint),
    }
    protocol["train"] = generate_split(
        model,
        data,
        noise_bank,
        args.output_dir,
        "train",
        args.num_train,
        ranges["train"],
        args.height,
        args.width,
        args.seed,
        device,
        args.batch_size,
        None,
        args.split_axis,
    )
    protocol["val"] = generate_split(
        model,
        data,
        noise_bank,
        args.output_dir,
        "val",
        args.num_val,
        ranges["val"],
        args.height,
        args.width,
        args.seed + 9000,
        device,
        args.batch_size,
        0.0,
        args.split_axis,
    )
    protocol["test"] = {}
    for snr, tag in ((-10.0, "m10"), (0.0, "0"), (10.0, "p10")):
        protocol["test"][tag] = generate_split(
            model,
            data,
            noise_bank,
            args.output_dir,
            f"test_{tag}",
            args.num_test,
            ranges["test"],
            args.height,
            args.width,
            args.seed + 18000,
            device,
            args.batch_size,
            snr,
            args.split_axis,
        )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
