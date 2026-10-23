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
import torch.utils.data as tdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SCRN-style 2D refiner")
    parser.add_argument("--scrn-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--input-mode",
        choices=["noisy_prior", "noisy"],
        default="noisy_prior",
    )
    parser.add_argument("--epochs", type=int, default=80)
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
    spec = importlib.util.spec_from_file_location("scrn_refiner", repo / "model" / "SCRN.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_one_channel_weights(model: nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = model.state_dict()
    for key, value in state.items():
        if key == "m_head.0.weight" and value.shape[1] == 1:
            adapted = torch.zeros_like(target[key])
            adapted[:, :1] = value
            target[key] = adapted
        elif key in target and target[key].shape == value.shape:
            target[key] = value
    model.load_state_dict(target)


def load_matching_weights(model: nn.Module, checkpoint: Path) -> None:
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )


def snr(clean: torch.Tensor, estimate: torch.Tensor) -> float:
    clean = clean.double()
    estimate = estimate.double()
    return float(
        10.0
        * torch.log10(
            (torch.sum(clean**2) + 1e-12)
            / (torch.sum((clean - estimate) ** 2) + 1e-12)
        )
    )


class SeisPACESectionDataset(tdata.Dataset):
    def __init__(
        self,
        root: Path,
        prefix: str,
        anchor: np.ndarray | None = None,
        input_mode: str = "noisy_prior",
    ) -> None:
        self.clean = np.load(root / f"{prefix}_clean.npy")
        self.noisy = np.load(root / f"{prefix}_noisy.npy")
        self.prior = np.load(root / f"{prefix}_prior.npy")
        self.anchor = anchor
        self.input_mode = input_mode

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, index: int):
        if self.input_mode == "noisy":
            inputs = self.noisy[index][None]
        else:
            inputs = np.stack([self.noisy[index], self.prior[index]])
        clean = torch.from_numpy(self.clean[index].astype(np.float32))[None]
        if self.anchor is None:
            return torch.from_numpy(inputs.astype(np.float32)), clean
        return (
            torch.from_numpy(inputs.astype(np.float32)),
            clean,
            torch.from_numpy(self.anchor[index].astype(np.float32))[None],
        )


def compute_anchor(model: nn.Module, root: Path, prefix: str, device: torch.device) -> np.ndarray:
    noisy = np.load(root / f"{prefix}_noisy.npy")
    output = np.empty_like(noisy)
    model.eval()
    with torch.no_grad():
        for start in range(0, noisy.shape[0], 16):
            stop = min(noisy.shape[0], start + 16)
            batch = torch.from_numpy(noisy[start:stop].astype(np.float32))[:, None].to(device)
            output[start:stop] = model(batch).cpu().numpy()[:, 0].astype(np.float16)
    return output


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    in_channels = 2 if args.input_mode == "noisy_prior" else 1
    scrn = load_scrn_module(args.scrn_root.resolve()).SCRN(in_nc=in_channels)
    scrn.m_tail[0] = nn.Conv2d(scrn.dim, 1, 3, 1, 1, bias=False)
    if args.resume_checkpoint is not None:
        load_matching_weights(scrn, args.resume_checkpoint)
    elif args.init_checkpoint is not None:
        if args.input_mode == "noisy_prior":
            load_one_channel_weights(scrn, args.init_checkpoint)
        else:
            load_matching_weights(scrn, args.init_checkpoint)
    model = scrn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 40, 60], gamma=0.2
    )
    criterion = nn.MSELoss()
    anchor_model = None
    train_anchor = None
    val_anchor = None
    if args.anchor_weight > 0:
        if args.anchor_checkpoint is None:
            raise ValueError("--anchor-checkpoint is required when anchor-weight > 0")
        anchor_model = load_scrn_module(
            args.scrn_root.resolve()
        ).SCRN().to(device)
        anchor_model.load_state_dict(
            torch.load(args.anchor_checkpoint, map_location=device, weights_only=False)
        )
        train_anchor = compute_anchor(anchor_model, args.data_dir, "train", device)
        val_anchor = compute_anchor(anchor_model, args.data_dir, "val", device)
    train_data = SeisPACESectionDataset(
        args.data_dir, "train", train_anchor, args.input_mode
    )
    val_data = SeisPACESectionDataset(args.data_dir, "val", val_anchor, args.input_mode)
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
        "method": "SeisPACE-2D",
        "inputs": ["noisy_section", "prior2d_prediction"]
        if args.input_mode == "noisy_prior"
        else ["noisy_section"],
        "input_mode": args.input_mode,
        "initialization": str(args.init_checkpoint)
        if args.init_checkpoint is not None
        else "random",
        "resume_checkpoint": str(args.resume_checkpoint)
        if args.resume_checkpoint is not None
        else None,
        "anchor_checkpoint": str(args.anchor_checkpoint)
        if args.anchor_checkpoint is not None
        else None,
        "anchor_weight": args.anchor_weight,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
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
        for batch in train_loader:
            inputs, clean = batch[0], batch[1]
            inputs = inputs.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            estimate = model(inputs)
            loss = criterion(estimate, clean)
            if args.anchor_weight > 0:
                anchor = batch[2].to(device, non_blocking=True)
                loss = loss + args.anchor_weight * criterion(estimate, anchor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        values = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, clean = batch[0], batch[1]
                inputs = inputs.to(device)
                clean = clean.to(device)
                estimate = model(inputs)
                for index in range(clean.shape[0]):
                    values.append(snr(clean[index], estimate[index]))
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
        print(
            f"epoch={epoch} loss={row['loss']:.6f} "
            f"val_snr={val_snr:.3f} seconds={row['seconds']:.2f}",
            flush=True,
        )
    print(f"best_validation_snr={best_snr:.3f}", flush=True)


if __name__ == "__main__":
    main()
