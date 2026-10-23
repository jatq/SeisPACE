from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from seispace.data import CSVDataset
from seispace.models import (
    CALIBRATOR_1D_CONFIG,
    CALIBRATOR_2D_CONFIG,
    PRIOR_1D_CONFIG,
    PRIOR_2D_CONFIG,
    adapt_baseline_for_prior,
    build_calibrator,
    build_prior,
    load_prior_checkpoint,
    state_dict_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SeisPACE prior, direct baseline, or calibrator."
    )
    parser.add_argument("--dimension", choices=["1d", "2d"], required=True)
    parser.add_argument(
        "--stage",
        choices=["prior", "baseline", "calibrate"],
        required=True,
    )
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--anchor-weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
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


def prior_batch(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    dimension: str,
) -> torch.Tensor:
    if dimension == "2d":
        return model.denoise(noisy)
    predictions = [
        model.predict_signal(trace[0].detach().cpu().numpy())
        for trace in noisy
    ]
    return torch.from_numpy(np.stack(predictions)[:, None]).to(noisy.device)


def estimate(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    output = model(inputs)
    return output[0] if isinstance(output, tuple) else output


def save_checkpoint(
    path: Path,
    dimension: str,
    stage: str,
    model: torch.nn.Module,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "seispace-checkpoint-v1",
            "dimension": dimension,
            "stage": stage,
            "config": config,
            "state_dict": model.state_dict(),
        },
        path,
    )
    print(f"saved {path}", flush=True)


def save_bundle(
    path: Path,
    dimension: str,
    prior: torch.nn.Module,
    prior_config: dict,
    calibrator: torch.nn.Module,
    calibrator_config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "seispace-bundle-v1",
            "version": "0.2.0",
            dimension: {
                "prior": {
                    "config": prior_config,
                    "state_dict": prior.state_dict(),
                },
                "calibrator": {
                    "config": calibrator_config,
                    "state_dict": calibrator.state_dict(),
                },
            },
        },
        path,
    )
    print(f"saved {path}", flush=True)


def train_prior(args: argparse.Namespace, device: torch.device) -> None:
    config = PRIOR_1D_CONFIG if args.dimension == "1d" else PRIOR_2D_CONFIG
    dataset = CSVDataset(
        args.train_csv,
        dimension=args.dimension,
        task="prior" if args.dimension == "1d" else "pair",
        patch_length=int(PRIOR_1D_CONFIG["patch_length"]),
        overlap_ratio=float(PRIOR_1D_CONFIG["overlap_ratio"]),
        num_segments=int(PRIOR_1D_CONFIG["num_segments"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_prior(args.dimension, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            if args.dimension == "1d":
                noisy, target = batch
                noisy = noisy.to(device)
                target = target.to(device)
                loss = model.trainloss(noisy, target, loss_type="mae")
            else:
                batch = batch.to(device)
                noisy, clean = batch[:, :1], batch[:, 1:2]
                loss = model.compute_loss(noisy, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"stage=prior epoch={epoch} loss={np.mean(losses):.6f}",
            flush=True,
        )
    save_checkpoint(args.output, args.dimension, "prior", model, config)


def train_calibrator(args: argparse.Namespace, device: torch.device) -> None:
    calibrate = args.stage == "calibrate"
    if calibrate and (
        args.prior_checkpoint is None or args.baseline_checkpoint is None
    ):
        raise ValueError(
            "--prior-checkpoint and --baseline-checkpoint are required "
            "for calibration"
        )
    if calibrate and args.anchor_weight < 0:
        raise ValueError("--anchor-weight must be non-negative")

    dataset = CSVDataset(
        args.train_csv,
        dimension=args.dimension,
        task="pair",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = (
        CALIBRATOR_1D_CONFIG if args.dimension == "1d" else CALIBRATOR_2D_CONFIG
    )
    model = build_calibrator(
        args.dimension,
        in_channels=2 if calibrate else 1,
        config=config,
    ).to(device)

    prior_model = None
    prior_config = None
    anchor_model = None
    if calibrate:
        prior_checkpoint = torch.load(
            args.prior_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        prior_config = (
            prior_checkpoint.get("config")
            if isinstance(prior_checkpoint, dict)
            else None
        )
        prior_config = prior_config or (
            PRIOR_1D_CONFIG if args.dimension == "1d" else PRIOR_2D_CONFIG
        )
        prior_model = load_prior_checkpoint(
            args.dimension,
            str(args.prior_checkpoint),
            device,
        )
        baseline_state = state_dict_from_checkpoint(
            str(args.baseline_checkpoint),
            device,
        )
        adapt_baseline_for_prior(model, baseline_state)
        anchor_model = build_calibrator(
            args.dimension,
            in_channels=1,
            config=config,
        ).to(device)
        anchor_model.load_state_dict(baseline_state)
        anchor_model.eval()
        for parameter in anchor_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            noisy, clean = batch[:, :1], batch[:, 1:2]
            if calibrate:
                with torch.no_grad():
                    prior = prior_batch(prior_model, noisy, args.dimension)
                inputs = torch.cat([noisy, prior], dim=1)
            else:
                inputs = noisy
            prediction = estimate(model, inputs)
            loss = F.mse_loss(prediction, clean)
            if anchor_model is not None:
                with torch.no_grad():
                    anchor = estimate(anchor_model, noisy)
                loss = loss + args.anchor_weight * F.mse_loss(prediction, anchor)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(
            f"stage={args.stage} epoch={epoch} loss={np.mean(losses):.6f}",
            flush=True,
        )
    if calibrate:
        save_bundle(
            args.output,
            args.dimension,
            prior_model,
            prior_config,
            model,
            config,
        )
    else:
        save_checkpoint(args.output, args.dimension, args.stage, model, config)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.stage == "prior":
        train_prior(args, device)
    else:
        train_calibrator(args, device)


if __name__ == "__main__":
    main()
