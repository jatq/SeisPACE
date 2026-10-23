from __future__ import annotations

from typing import Any

import torch

from .calibrator1d import SeisPACECalibrator1D
from .calibrator2d import SeisPACECalibrator2D
from .config import TransformConfig
from .prior1d import DualSparsePrior1D
from .prior2d import DualSparsePrior2D


PRIOR_1D_CONFIG: dict[str, Any] = {
    "embed_dim": 256,
    "patch_length": 256,
    "overlap_ratio": 0.703125,
    "num_segments": 15,
    "base_channels": [32, 64, 128, 256, 128, 64, 32, 16],
    "inference_normalization": "window",
    "embed_loss_weight": 0.1,
}

CALIBRATOR_1D_CONFIG: dict[str, Any] = {
    "channels": [32, 64, 128, 256],
}

PRIOR_2D_CONFIG: dict[str, Any] = {
    "patch_height": 16,
    "patch_width": 16,
    "stride_height": 8,
    "stride_width": 8,
    "embed_dim": 128,
    "dae_channels": [64, 32, 16, 8],
    "init_embedding": True,
    "embed_loss_weight": 0.1,
}

CALIBRATOR_2D_CONFIG: dict[str, Any] = {
    "config": [1, 1, 1, 1, 1],
    "dim": 64,
    "head_dim": 32,
    "window_size": 8,
    "input_resolution": 256,
}


def build_prior(
    dimension: str,
    config: dict[str, Any] | None = None,
) -> DualSparsePrior1D | DualSparsePrior2D:
    values = dict(PRIOR_1D_CONFIG if dimension == "1d" else PRIOR_2D_CONFIG)
    values.update(config or {})
    if dimension == "1d":
        transform = TransformConfig(
            patch_length=int(values["patch_length"]),
            overlap_ratio=float(values["overlap_ratio"]),
            num_segments=int(values["num_segments"]),
        )
        return DualSparsePrior1D(
            embed_dim=int(values["embed_dim"]),
            transform_config=transform,
            base_channels=tuple(values["base_channels"]),
            inference_normalization=str(values["inference_normalization"]),
            embed_loss_weight=float(values["embed_loss_weight"]),
        )
    return DualSparsePrior2D(
        patch_height=int(values["patch_height"]),
        patch_width=int(values["patch_width"]),
        stride_height=int(values["stride_height"]),
        stride_width=int(values["stride_width"]),
        embed_dim=int(values["embed_dim"]),
        dae_channels=tuple(values["dae_channels"]),
        init_embedding=bool(values["init_embedding"]),
        embed_loss_weight=float(values["embed_loss_weight"]),
    )


def build_calibrator(
    dimension: str,
    in_channels: int,
    config: dict[str, Any] | None = None,
) -> SeisPACECalibrator1D | SeisPACECalibrator2D:
    values = dict(
        CALIBRATOR_1D_CONFIG if dimension == "1d" else CALIBRATOR_2D_CONFIG
    )
    values.update(config or {})
    if dimension == "1d":
        return SeisPACECalibrator1D(
            channels=tuple(values["channels"]),
            in_channels=in_channels,
            residual=False,
            zero_head=False,
        )
    return SeisPACECalibrator2D(
        in_channels=in_channels,
        config=tuple(values["config"]),
        dim=int(values["dim"]),
        head_dim=int(values["head_dim"]),
        window_size=int(values["window_size"]),
        input_resolution=int(values["input_resolution"]),
    )


def state_dict_from_checkpoint(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} does not contain a state dictionary")
    return checkpoint


def load_prior_checkpoint(
    dimension: str,
    path: str,
    device: torch.device,
) -> DualSparsePrior1D | DualSparsePrior2D:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    model = build_prior(dimension, config)
    model.load_state_dict(state)
    return model.to(device).eval()


def adapt_baseline_for_prior(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    target = model.state_dict()
    input_layer = "stem.0.weight" if hasattr(model, "stem") else "m_head.0.weight"
    for key, value in state.items():
        if key == input_layer:
            adapted = torch.zeros_like(target[key])
            adapted[:, :1] = value[:, :1]
            target[key] = adapted
        elif key in target and target[key].shape == value.shape:
            target[key] = value
    model.load_state_dict(target)
