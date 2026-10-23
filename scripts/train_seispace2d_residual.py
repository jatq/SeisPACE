from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.prior_refiner import PriorResidualRefiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a zero-initialized 2D residual refiner"
    )
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_scrn_module(repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("scrn_residual", repo / "model" / "SCRN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeisPACESectionDataset(tdata.Dataset):
    def __init__(self, root: Path, prefix: str) -> None:
        self.clean = np.load(root / f"{prefix}_clean.npy")
        self.noisy = np.load(root / f"{prefix}_noisy.npy")
        self.prior = np.load(root / f"{prefix}_prior.npy")

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(
                np.stack([self.noisy[index], self.prior[index]]).astype(np.float32)
            ),
            torch.from_numpy(self.clean[index].astype(np.float32))[None],
            torch.from_numpy(self.prior[index].astype(np.float32))[None],
        )


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
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    module = load_scrn_module(args.scrn_root.resolve())
    backbone = module.SCRN(in_nc=2)
    backbone.m_tail[0] = nn.Conv2d(
        backbone.dim, 1, 3, 1, 1, bias=False
    )
    nn.init.zeros_(backbone.m_tail[0].weight)
    model = PriorResidualRefiner(backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.2
    )
    train_data = SeisPACESectionDataset(args.data_dir, "train")
    val_data = SeisPACESectionDataset(args.data_dir, "val")
    train_loader = tdata.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = tdata.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": "2D dual-sparse prior + zero-initialized residual refiner",
        "initialization": "random backbone; zero correction head",
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "output_ensemble": False,
        "external_baseline_weights": False,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=["epoch", "loss", "val_snr_db", "seconds"]
        ).writeheader()
    best_snr = -float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for inputs, clean, prior in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            estimate, correction = model(inputs)
            loss = F.mse_loss(estimate, clean) + 0.5 * F.l1_loss(
                correction, clean - prior
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        values = []
        with torch.no_grad():
            for inputs, clean, _ in val_loader:
                estimate, _ = model(inputs.to(device))
                for index in range(clean.shape[0]):
                    values.append(section_snr(clean[index], estimate[index].cpu()))
        val_snr = float(np.mean(values))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_snr_db": val_snr,
            "seconds": float(time.time() - started),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if val_snr > best_snr:
            best_snr = val_snr
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} loss={row['loss']:.6f} "
                f"val_snr={val_snr:.3f} seconds={row['seconds']:.2f}",
                flush=True,
            )
    print(f"best_validation_snr={best_snr:.3f}", flush=True)


if __name__ == "__main__":
    main()
