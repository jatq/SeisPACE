from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.utils.data as tdata
from scipy.io import loadmat

from .config import ExperimentConfig
from .metrics import energy_snr, snr_gain
from .prior1d import DualSparsePrior1D
from .transforms import mix_at_snr, normalize_std


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: ExperimentConfig,
    epoch: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "loss": loss,
        },
        path,
    )


def load_dual_sparse_prior(
    checkpoint: str | Path,
    config: ExperimentConfig,
    device: torch.device,
) -> DualSparsePrior1D:
    model = DualSparsePrior1D(
        embed_dim=config.model.embed_dim,
        transform_config=config.transform,
        base_channels=config.model.base_channels,
        inference_normalization=config.data.inference_normalization,
        embed_loss_weight=config.model.embed_loss_weight,
    )
    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload)
    model.load_compatible_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def train_model(
    model: torch.nn.Module,
    loader: tdata.DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    loss_fn: Callable[[torch.nn.Module, object, object], torch.Tensor] | None = None,
) -> list[dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(output_dir / "config.json")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.scheduler_step,
        gamma=config.train.scheduler_gamma,
    )
    use_amp = config.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    for epoch in range(config.train.epochs):
        model.train()
        losses: list[float] = []
        for step, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if loss_fn is None:
                    loss = model.trainloss(inputs, targets, loss_type=config.train.loss)
                else:
                    loss = loss_fn(model, inputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            if config.train.log_interval > 0 and (step + 1) % config.train.log_interval == 0:
                recent = np.mean(losses[-config.train.log_interval :])
                print(f"epoch={epoch} step={step + 1}/{len(loader)} loss={recent:.6f}")

        scheduler.step()
        epoch_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"epoch={epoch} loss={epoch_loss:.6f}")
        save_checkpoint(output_dir / "last.pt", model, config, epoch, epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(output_dir / "best.pt", model, config, epoch, epoch_loss)
        _write_history(output_dir / "history.csv", history)
    return history


def _load_wavelet_test(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(str(path))
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    return signals, noises


def evaluate_wavelet(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    test_file: str | Path,
    snr_values: list[float],
    max_signals: int | None = None,
) -> dict[str, list[float]]:
    signals, noises = _load_wavelet_test(test_file)
    results: dict[str, list[float]] = {}
    limit = signals.shape[0] if max_signals is None else min(signals.shape[0], max_signals)
    for snr in snr_values:
        gains: list[float] = []
        for index in range(limit):
            clean, noisy = mix_at_snr(signals[index], noises[index % noises.shape[0]], snr)
            denoised = predict_fn(noisy)
            gains.append(snr_gain(clean, noisy, denoised))
        key = f"{snr:g}"
        results[key] = gains
        print(f"snr={snr:g} dB mean_gain={np.mean(gains):.2f} dB")
    return results


def save_wavelet_results(
    results: dict[str, list[float]],
    path: str | Path,
    model_name: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        keys = list(results.keys())
        writer = csv.writer(handle)
        writer.writerow(["model", *keys])
        writer.writerow([model_name, *[f"{np.mean(results[key]):.2f}" for key in keys]])


def evaluate_stead_traces(
    model: DualSparsePrior1D,
    dataset: tdata.Dataset,
    max_traces: int | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    limit = len(dataset) if max_traces is None else min(len(dataset), max_traces)
    for index in range(limit):
        item = dataset[index]
        signal = np.asarray(item["signal"], dtype=np.float32)
        denoised = model.predict_signal(signal)
        before = energy_snr(signal, item["p_sample"], item["coda_sample"])
        after = energy_snr(denoised, item["p_sample"], item["coda_sample"])
        rows.append(
            {
                "trace_name": str(item["trace_name"]),
                "snr_before_db": before,
                "snr_after_db": after,
                "snr_gain_db": after - before,
            }
        )
        if (index + 1) % 100 == 0:
            print(f"evaluated={index + 1}/{limit}")
    return rows


def save_stead_results(rows: list[dict[str, float | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
