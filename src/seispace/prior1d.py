from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from .config import TransformConfig
from .transforms import ist_dct, segment_inputs, st_dct


def _remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map historical 2023 checkpoint keys to the current module names."""
    if not any(key.startswith("sparse_map.") for key in state_dict):
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("sparse_map."):
            suffix = key[len("sparse_map.") :]
            suffix = suffix.replace(".weight", ".conv.weight").replace(
                ".bias", ".conv.bias"
            )
            remapped[f"sparse_mapping.{suffix}"] = value
        elif key.startswith("unet."):
            suffix = key[len("unet.") :].replace(".deconv.", ".conv.")
            remapped[f"sparse_denoising.{suffix}"] = value
        elif key.startswith("head.0."):
            remapped[f"sparse_denoising.8.conv.{key[len('head.0.'):]}"] = value
        elif key.startswith("head.2."):
            remapped[f"inverse_sparse_mapping.1.conv.{key[len('head.2.'):]}"] = value
        else:
            remapped[key] = value
    return remapped


def _remap_upstream_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map the released 1D prior checkpoint keys to the current module names."""
    if "embed.weight" not in state_dict:
        return state_dict
    remapped: dict[str, torch.Tensor] = {}
    ignored = {"dct_weight", "idct_weight", "identity"}
    for key, value in state_dict.items():
        if key in ignored:
            continue
        if key.startswith("embed."):
            remapped[f"sparse_mapping.1.conv.{key[len('embed.'):]}"] = value
        elif key.startswith("inverse_embed."):
            remapped[
                f"inverse_sparse_mapping.1.conv.{key[len('inverse_embed.'):]}"
            ] = value
        elif key.startswith("dae.unet."):
            suffix = key[len("dae.unet.") :].replace(".deconv.", ".conv.")
            if suffix.startswith("8.") and not suffix.startswith("8.conv."):
                suffix = "8.conv." + suffix[2:]
            remapped[f"sparse_denoising.{suffix}"] = value
        else:
            remapped[key] = value
    return remapped


def _drop_embedding_bias(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    ignored = {
        "sparse_mapping.1.conv.bias",
        "inverse_sparse_mapping.1.conv.bias",
    }
    return {key: value for key, value in state_dict.items() if key not in ignored}


class PWC(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CBR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DBR(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Squeeze(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class Transpose(nn.Module):
    def __init__(self, dims: tuple[int, int] = (1, 2)) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.dims)


class DualSparsePrior1D(nn.Module):
    """Lightweight dual-sparse seismic prior with legacy checkpoint support.

    Historical checkpoint keys are translated by ``load_compatible_state_dict``;
    new checkpoints use the current SeisPACE module names.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        transform_config: TransformConfig | None = None,
        base_channels: Sequence[int] = (32, 64, 128, 256, 128, 64, 32, 16),
        inference_normalization: str = "window",
        embed_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.transform_config = transform_config or TransformConfig()
        self.embed_dim = int(embed_dim)
        self.inference_normalization = inference_normalization
        self.embed_loss_weight = float(embed_loss_weight)
        self.patch_length = self.transform_config.patch_length
        self.num_segments = self.transform_config.num_segments

        self.sparse_mapping = nn.Sequential(
            Transpose(dims=(1, 2)),
            PWC(self.patch_length, self.embed_dim, bias=False),
            Transpose(dims=(1, 2)),
        )

        self.shortcut_index = [np.array([0, 1, 2]), np.array([5, 6, 7])]
        channels = [self.num_segments] + list(base_channels)
        shortcut_channels: list[int] = []
        self.sparse_denoising = nn.ModuleList()
        for index, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            if index in self.shortcut_index[0]:
                shortcut_channels.append(out_channels)
            extra = shortcut_channels.pop() if index in self.shortcut_index[1] else 0
            layer_cls = DBR if in_channels // out_channels == 2 else CBR
            self.sparse_denoising.append(layer_cls(in_channels + extra, out_channels))
        self.sparse_denoising.append(PWC(channels[-1], 1))

        self.inverse_sparse_mapping = nn.Sequential(
            Transpose(dims=(1, 2)),
            PWC(self.embed_dim, self.patch_length, bias=False),
            Transpose(dims=(1, 2)),
            Squeeze(dim=1),
        )

        self.mae = nn.L1Loss(reduction="mean")
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                "Expected inputs with shape (batch, windows, segments, patch_length)"
            )
        batch, windows, segments, patch_length = inputs.shape
        if segments != self.num_segments or patch_length != self.patch_length:
            raise ValueError(
                f"Expected ({self.num_segments}, {self.patch_length}) segments, "
                f"got ({segments}, {patch_length})"
            )

        x = inputs.reshape(-1, self.num_segments, self.patch_length)
        x = self.sparse_mapping(x).reshape(-1, self.num_segments, self.embed_dim)

        shortcuts: list[torch.Tensor] = []
        for index, layer in enumerate(self.sparse_denoising):
            x = layer(x)
            if index in self.shortcut_index[0]:
                shortcuts.append(x)
            elif index in self.shortcut_index[1] - 1:
                x = torch.cat([x, shortcuts.pop()], dim=1)
        return self.inverse_sparse_mapping(x)

    def trainloss(
        self, inputs: torch.Tensor, targets: torch.Tensor, loss_type: str = "mae"
    ) -> torch.Tensor:
        predictions = self.forward(inputs)
        targets = targets.reshape(-1, self.patch_length)
        if loss_type == "mae":
            loss = self.mae(predictions, targets)
        elif loss_type == "mse":
            loss = self.mse(predictions, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        if self.embed_loss_weight > 0:
            analysis = self.sparse_mapping[1].conv.weight[:, :, 0]
            synthesis = self.inverse_sparse_mapping[1].conv.weight[:, :, 0]
            if analysis.shape[0] == analysis.shape[1]:
                target = torch.eye(
                    analysis.shape[0], device=analysis.device, dtype=analysis.dtype
                )
                product = analysis @ synthesis
            else:
                target = torch.zeros(
                    analysis.shape[1], analysis.shape[1],
                    device=analysis.device,
                    dtype=analysis.dtype,
                )
                retained = min(analysis.shape[0], analysis.shape[1])
                target[:retained, :retained] = torch.eye(
                    retained, device=analysis.device, dtype=analysis.dtype
                )
                product = synthesis @ analysis
            loss = loss + self.embed_loss_weight * torch.mean((product - target) ** 2)
        return loss

    @torch.no_grad()
    def predict_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            return self._predict_1d(signal)
        return np.stack([self._predict_1d(trace) for trace in signal], axis=0)

    def _predict_1d(self, signal: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        if self.inference_normalization == "trace":
            mean = float(np.mean(signal))
            scale = float(max(np.std(signal), 1e-8))
            normalized_signal = (signal - mean) / scale
            windows = st_dct(normalized_signal, self.transform_config)
            inputs = segment_inputs(windows, self.transform_config)
            tensor = torch.from_numpy(inputs.astype(np.float32)).to(device)
            predictions = self.forward(tensor).cpu().numpy() * scale
        elif self.inference_normalization == "window":
            windows = st_dct(signal, self.transform_config)
            inputs = segment_inputs(windows, self.transform_config)[0]
            scale = np.std(inputs, axis=(-2, -1), keepdims=True)
            scale = np.maximum(scale, 1e-8)
            normalized = inputs / scale
            tensor = (
                torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(device)
            )
            predictions = self.forward(tensor).cpu().numpy() * scale.reshape(-1, 1)
        else:
            raise ValueError(
                f"Unknown inference_normalization: {self.inference_normalization}"
            )
        denoised = ist_dct(predictions[None, ...], self.transform_config)[0]
        if denoised.size < signal.size:
            denoised = np.pad(denoised, (0, signal.size - denoised.size), mode="constant")
        return denoised[: signal.size].astype(np.float32)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def load_compatible_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = True
    ):
        return super().load_state_dict(
            _drop_embedding_bias(
                _remap_upstream_state_dict(_remap_legacy_state_dict(state_dict))
            ),
            strict=strict,
        )
