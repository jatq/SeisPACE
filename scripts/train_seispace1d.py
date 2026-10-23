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
    parser = argparse.ArgumentParser(description="Train SeisPACE 1D ablations")
    parser.add_argument(
        "--mode",
        choices=["baseline", "calibrate", "calibrate_scratch", "noisy_anchor"],
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--refiner-checkpoint", type=Path)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int)
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
            torch.from_numpy(
                np.stack([self.noisy[index], self.prior[index]]).astype(np.float32)
            ),
            torch.from_numpy(self.clean[index])[None],
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


def adapt_baseline_to_calibrator(model: SeisPACECalibrator1D, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = model.state_dict()
    for key, value in state.items():
        if key == "stem.0.weight":
            adapted = torch.zeros_like(target[key])
            adapted[:, :1] = value
            target[key] = adapted
        elif key in target and target[key].shape == value.shape:
            target[key] = value
    model.load_state_dict(target)


def load_one_channel_baseline(model: SeisPACECalibrator1D, checkpoint: Path) -> None:
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )


def load_matching_weights(model: SeisPACECalibrator1D, checkpoint: Path) -> None:
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["train"]["seed"])
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    channels = tuple(config["model"]["channels"])
    use_prior = args.mode in {"calibrate", "calibrate_scratch"}
    if args.mode == "baseline":
        if args.anchor_weight > 0:
            raise ValueError("baseline mode does not use an anchor loss")
        model = SeisPACECalibrator1D(
            channels=channels,
            in_channels=1,
            residual=False,
            zero_head=False,
        ).to(device)
        init_checkpoint = args.init_checkpoint or args.refiner_checkpoint
        if init_checkpoint is not None:
            load_matching_weights(model, init_checkpoint)
            initial_state = str(init_checkpoint)
        else:
            initial_state = "random"
    else:
        if args.mode == "calibrate":
            model = SeisPACECalibrator1D(
                channels=channels,
                in_channels=2,
                residual=False,
                zero_head=False,
            ).to(device)
            init_checkpoint = args.init_checkpoint or args.refiner_checkpoint
            if init_checkpoint is not None:
                load_matching_weights(model, init_checkpoint)
                initial_state = str(init_checkpoint)
            else:
                if args.baseline_checkpoint is None:
                    raise ValueError(
                        "calibration mode requires --baseline-checkpoint or "
                        "--refiner-checkpoint"
                    )
                adapt_baseline_to_calibrator(model, args.baseline_checkpoint)
                initial_state = "same-data 1D baseline with zeroed prior channel"
        elif args.mode == "calibrate_scratch":
            model = SeisPACECalibrator1D(
                channels=channels,
                in_channels=2,
                residual=False,
                zero_head=False,
            ).to(device)
            if args.init_checkpoint is not None:
                load_matching_weights(model, args.init_checkpoint)
                initial_state = str(args.init_checkpoint)
            else:
                initial_state = "random"
        else:
            if args.baseline_checkpoint is None:
                raise ValueError(
                    "noisy_anchor mode requires --baseline-checkpoint"
                )
            model = SeisPACECalibrator1D(
                channels=channels,
                in_channels=1,
                residual=False,
                zero_head=False,
            ).to(device)
            init_checkpoint = args.init_checkpoint or args.refiner_checkpoint
            if init_checkpoint is not None:
                load_matching_weights(model, init_checkpoint)
                initial_state = str(init_checkpoint)
            else:
                load_one_channel_baseline(model, args.baseline_checkpoint)
                initial_state = "same-data 1D baseline"

    if args.anchor_weight > 0:
        if args.baseline_checkpoint is None:
            raise ValueError("--baseline-checkpoint is required for anchor training")
        anchor_model = SeisPACECalibrator1D(
            channels=channels,
            in_channels=1,
            residual=False,
            zero_head=False,
        ).to(device)
        anchor_model.load_state_dict(
            torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        )
        anchor_model.eval()
    else:
        anchor_model = None
    train_data = SeisPACETraceDataset(args.data_dir, "train")
    train_loader = tdata.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loaders = {
        name: tdata.DataLoader(
            SeisPACETraceDataset(args.data_dir, name),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for name in ("val_m10", "val_p0", "val_p10")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["train"]["scheduler_step"]),
        gamma=float(config["train"]["scheduler_gamma"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "method": f"SeisPACE 1D {args.mode}",
        "mode": args.mode,
        "inputs": ["noisy_trace", "dual_sparse_prior"]
        if use_prior
        else ["noisy_trace"],
        "initialization": initial_state,
        "baseline_checkpoint": str(args.baseline_checkpoint)
        if args.baseline_checkpoint is not None
        else None,
        "init_checkpoint": str(args.init_checkpoint)
        if args.init_checkpoint is not None
        else None,
        "refiner_checkpoint": str(args.refiner_checkpoint)
        if args.refiner_checkpoint is not None
        else None,
        "anchor_weight": args.anchor_weight,
        "train_samples": len(train_data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": seed,
        "channels": list(channels),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=["epoch", "loss", "val_mean_snr", "seconds"]
        ).writeheader()
    best_snr = -float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for inputs, clean in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            model_input = inputs if use_prior else inputs[:, :1]
            estimate, _ = model(model_input)
            loss = F.mse_loss(estimate, clean)
            if anchor_model is not None:
                with torch.no_grad():
                    anchor, _ = anchor_model(inputs[:, :1])
                loss = loss + args.anchor_weight * F.mse_loss(estimate, anchor)
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
                for inputs, clean in loader:
                    model_input = (
                        inputs.to(device) if use_prior else inputs[:, :1].to(device)
                    )
                    estimate, _ = model(model_input)
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
                f"epoch={epoch} loss={row['loss']:.6f} val_mean={val_mean:.3f} "
                f"levels={snr_by_level} seconds={row['seconds']:.2f}",
                flush=True,
            )
    print(f"best_validation_mean_snr={best_snr:.3f}", flush=True)


if __name__ == "__main__":
    main()
