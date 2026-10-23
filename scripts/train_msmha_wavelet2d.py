from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.baselines.msmha import MSMHA  # noqa: E402
from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain a PyTorch MSMHA reimplementation on wavelet sections"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat-file", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-sections", type=int, default=2000)
    parser.add_argument("--patches-per-section", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_patches(
    mat_file: Path,
    num_sections: int,
    patches_per_section: int,
    patch_size: int,
    seed: int,
) -> np.ndarray:
    sections = WaveletSectionDataset(
        mat_file,
        num_sections,
        training=True,
        snr_range=(-10.0, 10.0),
        seed=seed,
    )
    rng = np.random.default_rng(seed + 991)
    patches = np.empty(
        (num_sections * patches_per_section, patch_size, patch_size),
        dtype=np.float32,
    )
    cursor = 0
    for index in range(num_sections):
        _, noisy = sections[index]
        for _ in range(patches_per_section):
            row = int(rng.integers(0, noisy.shape[0] - patch_size + 1))
            column = int(rng.integers(0, noisy.shape[1] - patch_size + 1))
            patch = noisy[row : row + patch_size, column : column + patch_size]
            patches[cursor] = patch
            cursor += 1
        if (index + 1) % 200 == 0:
            print(f"prepared {index + 1}/{num_sections} sections", flush=True)
    return patches


def build_cached_patches(
    data_dir: Path,
    num_sections: int,
    patches_per_section: int,
    patch_size: int,
    seed: int,
) -> np.ndarray:
    noisy_sections = np.load(data_dir / "train_noisy.npy")
    if min(noisy_sections.shape[1:]) < patch_size:
        raise ValueError(
            f"Cached sections {noisy_sections.shape} are smaller than {patch_size}"
        )
    rng = np.random.default_rng(seed + 991)
    patches = np.empty(
        (num_sections * patches_per_section, patch_size, patch_size),
        dtype=np.float32,
    )
    cursor = 0
    for index in range(num_sections):
        noisy = noisy_sections[index % len(noisy_sections)]
        for _ in range(patches_per_section):
            row = int(rng.integers(0, noisy.shape[0] - patch_size + 1))
            column = int(rng.integers(0, noisy.shape[1] - patch_size + 1))
            patches[cursor] = noisy[
                row : row + patch_size, column : column + patch_size
            ]
            cursor += 1
        if (index + 1) % 200 == 0:
            print(f"prepared {index + 1}/{num_sections} cached sections", flush=True)
    return patches


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": "MSMHA PyTorch reimplementation",
        "source": "official DeepWave-KAUST/MSMHA Utils.py architecture",
        "training": "self-supervised identity loss on noisy wavelet-derived sections",
        "pretrained_checkpoint": None,
        "source_mat": str(args.mat_file) if args.mat_file is not None else None,
        "source_data_dir": str(args.data_dir)
        if args.data_dir is not None
        else None,
        "num_sections": args.num_sections,
        "patches_per_section": args.patches_per_section,
        "patch_size": args.patch_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    patch_args = (
        args.num_sections,
        args.patches_per_section,
        args.patch_size,
        args.seed,
    )
    patches = (
        build_patches(args.mat_file, *patch_args)
        if args.mat_file is not None
        else build_cached_patches(args.data_dir, *patch_args)
    )
    dataset = tdata.TensorDataset(torch.from_numpy(patches))
    loader = tdata.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = MSMHA(
        patch_height=args.patch_size,
        patch_width=args.patch_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    print(
        f"trainable parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=["epoch", "loss", "seconds"]
        ).writeheader()
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for (clean_patch,) in loader:
            target = clean_patch.to(device, non_blocking=True)
            first, second = model(target)
            loss = F.mse_loss(first, target) + F.mse_loss(second, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "seconds": float(time.time() - started),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch} loss={row['loss']:.6f} "
                f"seconds={row['seconds']:.2f}",
                flush=True,
            )
    torch.save(model.state_dict(), args.output_dir / "final.pt")
    print(f"saved {args.output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
