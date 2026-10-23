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

from seispace.calibrator1d import SeisPACECalibrator1D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 1D SeisPACE residual calibrator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeisPACETraceDataset(tdata.Dataset):
    def __init__(self, root: Path, prefix: str) -> None:
        self.clean = np.load(root / f"{prefix}_clean.npy")
        self.noisy = np.load(root / f"{prefix}_noisy.npy")
        self.prior = np.load(root / f"{prefix}_prior.npy")

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(np.stack([self.noisy[index], self.prior[index]])),
            torch.from_numpy(self.clean[index])[None],
            torch.from_numpy(self.prior[index])[None],
        )


def trace_snr(clean: torch.Tensor, estimate: torch.Tensor) -> float:
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
    set_seed(int(config["train"]["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_data = SeisPACETraceDataset(args.data_dir, "train")
    train_loader = tdata.DataLoader(
        train_data,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loaders = {}
    for name in ("val_m10", "val_p0", "val_p10"):
        val_loaders[name] = tdata.DataLoader(
            SeisPACETraceDataset(args.data_dir, name),
            batch_size=int(config["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
    model = SeisPACECalibrator1D(tuple(config["model"]["channels"])).to(device)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": "1D dual-sparse prior + residual SeisPACE calibrator",
        "initialization": "zero correction head; output equals the prior at initialization",
        "train_samples": len(train_data),
        "epochs": int(config["train"]["epochs"]),
        "batch_size": int(config["train"]["batch_size"]),
        "lr": float(config["train"]["lr"]),
        "seed": int(config["train"]["seed"]),
        **config["model"],
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
            handle, fieldnames=["epoch", "loss", "val_mean_snr", "seconds"]
        ).writeheader()
    best_snr = -float("inf")
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        started = time.time()
        model.train()
        losses = []
        for inputs, clean, prior in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            prior = prior.to(device, non_blocking=True)
            estimate, correction = model(inputs)
            target_correction = clean - prior
            loss = F.mse_loss(estimate, clean) + 0.5 * F.l1_loss(
                correction, target_correction
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        model.eval()
        snr_by_level = {}
        with torch.no_grad():
            for name, loader in val_loaders.items():
                values = []
                for inputs, clean, _ in loader:
                    estimate, _ = model(inputs.to(device))
                    for index in range(clean.shape[0]):
                        values.append(trace_snr(clean[index], estimate[index].cpu()))
                snr_by_level[name] = float(np.mean(values))
        val_mean = float(np.mean(list(snr_by_level.values())))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_mean_snr": val_mean,
            "seconds": float(time.time() - started),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        torch.save(model.state_dict(), args.output_dir / "last.pt")
        if val_mean > best_snr:
            best_snr = val_mean
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch} loss={row['loss']:.6f} "
                f"val_mean={val_mean:.3f} levels={snr_by_level} "
                f"seconds={row['seconds']:.2f}",
                flush=True,
            )
    print(f"best_validation_mean_snr={best_snr:.3f}", flush=True)


if __name__ == "__main__":
    main()
