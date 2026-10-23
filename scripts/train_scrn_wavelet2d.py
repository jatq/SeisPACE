from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain SCRN on wavelet-derived 2D data")
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--mat-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-train", type=int, default=2000)
    parser.add_argument("--num-val", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_scrn_module(repo: Path):
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("scrn_network", repo / "model" / "SCRN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    protocol = {
        "implementation": "javashs/SCRN",
        "initialization": "random; no pretrained checkpoint loaded",
        "data": str(args.mat_file),
        "train_samples": args.num_train,
        "validation_samples": args.num_val,
        "input_shape": [256, 128],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    scrn_module = load_scrn_module(args.scrn_root.resolve())
    model = scrn_module.SCRN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60], gamma=0.2
    )
    criterion = nn.MSELoss(reduction="mean")

    train_data = WaveletSectionDataset(
        args.mat_file,
        args.num_train,
        training=True,
        snr_range=(-10.0, 10.0),
        seed=args.seed,
    )
    val_data = WaveletSectionDataset(
        args.mat_file,
        args.num_val,
        training=False,
        snr_db=0.0,
        seed=args.seed + 10000,
    )
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

    history: list[dict[str, float]] = []
    best_snr = -float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for clean, noisy in train_loader:
            clean = clean.to(device, non_blocking=True).unsqueeze(1)
            noisy = noisy.to(device, non_blocking=True).unsqueeze(1)
            prediction = model(noisy)
            loss = criterion(prediction, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        snr_values = []
        with torch.no_grad():
            for clean, noisy in val_loader:
                clean = clean.to(device, non_blocking=True).unsqueeze(1)
                noisy = noisy.to(device, non_blocking=True).unsqueeze(1)
                prediction = model(noisy)
                snr_values.append(section_snr(clean, prediction))
        val_snr = float(np.mean(snr_values))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_snr_db": val_snr}
        history.append(row)
        print(
            f"epoch={epoch} loss={row['loss']:.6f} val_snr={val_snr:.2f} dB",
            flush=True,
        )
        torch.save(model.state_dict(), output_dir / "last.pt")
        if val_snr > best_snr:
            best_snr = val_snr
            torch.save(model.state_dict(), output_dir / "best.pt")
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    main()
