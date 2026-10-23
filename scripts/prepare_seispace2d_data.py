from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.prior2d import DualSparsePrior2D  # noqa: E402
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 2D dual-sparse prior calibration inputs")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_model(config: dict, checkpoint: Path, device: torch.device) -> DualSparsePrior2D:
    model_config = config["model"]
    model = DualSparsePrior2D(
        patch_height=int(model_config["patch_height"]),
        patch_width=int(model_config["patch_width"]),
        stride_height=int(model_config["stride_height"]),
        stride_width=int(model_config["stride_width"]),
        embed_dim=int(model_config["embed_dim"]),
        dae_channels=tuple(model_config["dae_channels"]),
        init_embedding=bool(model_config["init_embedding"]),
        embed_loss_weight=float(model_config["embed_loss_weight"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    return model


def generate_split(
    model: DualSparsePrior2D,
    mat_file: Path,
    count: int,
    training: bool,
    snr_db: float | None,
    seed: int,
    batch_size: int,
    device: torch.device,
    output_dir: Path,
    prefix: str,
) -> None:
    dataset = WaveletSectionDataset(
        mat_file,
        count,
        training=training,
        snr_db=snr_db,
        snr_range=(-10.0, 10.0),
        seed=seed,
    )
    clean_all = np.empty((count, 256, 128), dtype=np.float16)
    noisy_all = np.empty_like(clean_all)
    prior_all = np.empty_like(clean_all)
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            items = [dataset[index] for index in range(start, stop)]
            clean = torch.from_numpy(np.stack([item[0] for item in items]))[:, None]
            noisy = torch.from_numpy(np.stack([item[1] for item in items]))[:, None]
            prior = model.denoise(noisy.to(device)).cpu().numpy()[:, 0]
            clean_all[start:stop] = clean.numpy()[:, 0].astype(np.float16)
            noisy_all[start:stop] = noisy.numpy()[:, 0].astype(np.float16)
            prior_all[start:stop] = prior.astype(np.float16)
            print(f"{prefix}: prepared {stop}/{count}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{prefix}_clean.npy", clean_all)
    np.save(output_dir / f"{prefix}_noisy.npy", noisy_all)
    np.save(output_dir / f"{prefix}_prior.npy", prior_all)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.checkpoint, device)
    generate_split(
        model,
        args.mat_file,
        args.num_train,
        True,
        None,
        args.seed,
        args.batch_size,
        device,
        args.output_dir,
        "train",
    )
    generate_split(
        model,
        args.mat_file,
        args.num_val,
        False,
        0.0,
        args.seed + 9000,
        args.batch_size,
        device,
        args.output_dir,
        "val",
    )
    protocol = {
        "prior2d_config": str(args.config),
        "prior2d_checkpoint": str(args.checkpoint),
        "num_train": args.num_train,
        "num_val": args.num_val,
        "train_seed": args.seed,
        "validation_snr_db": 0.0,
        "storage_dtype": "float16",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
