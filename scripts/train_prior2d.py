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
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.prior2d import DualSparsePrior2D  # noqa: E402
from seispace.synthetic2d import (  # noqa: E402
    CachedSectionDataset,
    WaveletSectionDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 2D dual-sparse prior")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-sections", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def section_snr(clean: torch.Tensor, estimate: torch.Tensor) -> float:
    clean = clean.double()
    estimate = estimate.double()
    return float(
        10.0
        * torch.log10(
            (torch.sum(clean**2) + 1e-12)
            / (torch.sum((clean - estimate) ** 2) + 1e-12)
        )
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_sections is not None:
        config["train"]["num_sections"] = args.num_sections
    seed = int(config["train"]["seed"])
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir is not None:
        train_data = CachedSectionDataset(args.data_dir, "train")
        validation_data = CachedSectionDataset(args.data_dir, "val")
    else:
        if args.mat_file is None:
            raise ValueError("Either --mat-file or --data-dir is required")
        train_data = WaveletSectionDataset(
            args.mat_file,
            int(config["train"]["num_sections"]),
            training=True,
            snr_range=(-10.0, 10.0),
            seed=seed,
        )
        validation_data = WaveletSectionDataset(
            args.mat_file,
            int(config["train"]["num_validation"]),
            training=False,
            snr_db=0.0,
            seed=seed + 9000,
        )
    train_loader = tdata.DataLoader(
        train_data,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = tdata.DataLoader(
        validation_data,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
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
    if args.init_checkpoint is not None:
        model.load_state_dict(
            torch.load(
                args.init_checkpoint,
                map_location=device,
                weights_only=False,
            )
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["train"]["scheduler_step"]),
        gamma=float(config["train"]["scheduler_gamma"]),
    )
    protocol = {
        "method": "2D dual-sparse prior",
        "implementation": "two-dimensional DCT + learned embedding + 2D context DAE",
        "initialization": str(args.init_checkpoint)
        if args.init_checkpoint is not None
        else "from scratch; DCT projection embedding init",
        "source_mat": str(args.mat_file) if args.mat_file is not None else None,
        "source_data_dir": str(args.data_dir) if args.data_dir is not None else None,
        "training_sections": len(train_data),
        "validation_sections": len(validation_data),
        "validation_seed": seed + 9000,
        "epochs": int(config["train"]["epochs"]),
        "batch_size": int(config["train"]["batch_size"]),
        "lr": float(config["train"]["lr"]),
        "seed": seed,
        **model_config,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    print(
        f"trainable parameters={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle,
            fieldnames=["epoch", "loss", "val_snr_db", "learning_rate", "seconds"],
        ).writeheader()
    best_snr = -float("inf")
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        started = time.time()
        model.train()
        losses = []
        for clean, noisy in train_loader:
            clean = clean[:, None].to(device, non_blocking=True)
            noisy = noisy[:, None].to(device, non_blocking=True)
            loss = model.compute_loss(noisy, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_snr = float("nan")
        if (
            epoch % int(config["train"]["validation_interval"]) == 0
            or epoch == int(config["train"]["epochs"])
        ):
            model.eval()
            values = []
            with torch.no_grad():
                for clean, noisy in validation_loader:
                    clean_device = clean[:, None].to(device)
                    noisy_device = noisy[:, None].to(device)
                    estimate = model.denoise(noisy_device)
                    for index in range(clean.shape[0]):
                        values.append(
                            section_snr(clean_device[index], estimate[index])
                        )
            val_snr = float(np.mean(values))
            if val_snr > best_snr:
                best_snr = val_snr
                torch.save(model.state_dict(), args.output_dir / "best.pt")
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_snr_db": val_snr,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - started),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config["train"]["epochs"]):
            print(
                f"epoch={epoch} loss={row['loss']:.6f} "
                f"val_snr={row['val_snr_db']:.3f} seconds={row['seconds']:.2f}",
                flush=True,
            )
    torch.save(model.state_dict(), args.output_dir / "final.pt")
    print(
        f"saved {args.output_dir / 'final.pt'} best_validation_snr={best_snr:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
