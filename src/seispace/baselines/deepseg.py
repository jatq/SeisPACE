from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
from scipy.fftpack import dct, idct
from scipy.io import loadmat

from ..config import TransformConfig
from ..transforms import (
    mix_at_snr,
    normalize_std,
)


def _stack_windows_no_signal_pad(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    usable = ((x.shape[1] - cfg.patch_length) // cfg.stride) * cfg.stride + cfg.patch_length
    if usable < cfg.patch_length:
        raise ValueError("Signal is too short for DeepSeg")
    starts = np.arange(0, usable - cfg.patch_length + 1, cfg.stride)
    return np.stack(
        [x[:, start : start + cfg.patch_length] for start in starts], axis=1
    )


def stdct_deepseg(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    return dct(_stack_windows_no_signal_pad(x, cfg), norm=cfg.norm).astype(np.float32)


def _destack_windows_no_signal_pad(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim == 2:
        windows = windows[None, ...]
    channels, num_windows, patch_length = windows.shape
    output_length = patch_length + (num_windows - 1) * cfg.stride
    output = np.zeros((channels, output_length), dtype=np.float32)
    weight = np.zeros(output_length, dtype=np.float32)
    for index, window in enumerate(windows.transpose(1, 0, 2)):
        start = index * cfg.stride
        stop = start + patch_length
        output[:, start:stop] += window
        weight[start:stop] += 1.0
    return output / np.maximum(weight[None, :], 1.0)


def ist_dct_deepseg(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    samples = idct(np.asarray(windows, dtype=np.float32), norm=cfg.norm)
    return _destack_windows_no_signal_pad(samples, cfg)


def generate_inputs_from_targets(
    targets: np.ndarray, cfg: TransformConfig
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float32)
    if targets.ndim == 2:
        targets = targets[None, ...]
    pad = cfg.pad_segments
    zeros = np.zeros((targets.shape[0], pad, cfg.patch_length), dtype=np.float32)
    extended = np.concatenate([zeros, targets, zeros], axis=1)
    num_windows = extended.shape[1] - cfg.num_segments + 1
    windows = np.stack(
        [
            extended[:, start : start + cfg.num_segments, :]
            for start in range(num_windows)
        ],
        axis=1,
    )
    return windows


class Conv3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Deconv3x3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, output_padding: tuple[int, int]
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=output_padding,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DeepSeg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleList()
        shortcuts: list[int] = []
        encoder_channels = [8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512]
        encoder_strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        for index, (channels, stride) in enumerate(
            zip(encoder_channels, encoder_strides)
        ):
            self.encoder.append(
                Conv3x3(1 if index == 0 else encoder_channels[index - 1], channels, stride)
            )
            if index in [1, 2, 4, 6, 8, 10]:
                shortcuts.append(channels)

        decoder_channels = [512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 16, 8]
        decoder_strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        output_padding = [(1, 1), (1, 1), (1, 1), (1, 1), (0, 1), (0, 1)]
        self.decoder = nn.ModuleList()
        for index, (channels, stride) in enumerate(
            zip(decoder_channels, decoder_strides)
        ):
            if index == 0:
                self.decoder.append(Conv3x3(encoder_channels[-1], channels))
                continue
            extra = shortcuts.pop() if index in [2, 4, 6, 8, 10, 11] else 0
            in_channels = decoder_channels[index - 1] + extra
            if stride == 1:
                self.decoder.append(Conv3x3(in_channels, channels))
            else:
                self.decoder.append(
                    Deconv3x3(in_channels, channels, output_padding.pop())
                )
        self.conv_last = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)
        self.fc = nn.Linear(16 * 128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            batch, windows, channels, patch, features = x.shape
            x = x.reshape(batch * windows, channels, patch, features)
        shortcuts: list[torch.Tensor] = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in [1, 2, 4, 6, 8, 10]:
                shortcuts.append(x)
        for index, layer in enumerate(self.decoder):
            if index in [2, 4, 6, 8, 10, 11]:
                x = torch.cat([x, shortcuts.pop()], dim=1)
            x = layer(x)
        x = self.conv_last(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

    def trainloss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        if targets.ndim == 3:
            targets = targets.reshape(-1, targets.shape[-1])
        if loss_type == "mae":
            return nn.L1Loss(reduction="mean")(predictions, targets)
        return nn.MSELoss(reduction="mean")(predictions, targets)


class DeepSegWaveletDataset(tdata.Dataset):
    def __init__(
        self,
        path: str | Path,
        transform_config: TransformConfig,
        training: bool = True,
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        data = loadmat(str(path))
        self.signals = normalize_std(data["R"].T.astype(np.float32))[0]
        self.noises = normalize_std(data["z"].T.astype(np.float32))[0]
        self.transform_config = transform_config
        self.training = training
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        clean = self.signals[index]
        noise_index = (
            int(self.rng.integers(self.noises.shape[0]))
            if self.training
            else int(index % self.noises.shape[0])
        )
        snr = (
            float(self.rng.uniform(*self.snr_range))
            if self.snr_db is None
            else float(self.snr_db)
        )
        clean, noisy = mix_at_snr(clean, self.noises[noise_index], snr)
        noisy_dct = stdct_deepseg(noisy, self.transform_config)
        clean_dct = stdct_deepseg(clean, self.transform_config)
        mean = np.mean(noisy_dct)
        scale = max(float(np.std(noisy_dct)), 1e-8)
        noisy_dct = (noisy_dct - mean) / scale
        clean_dct = clean_dct / scale
        inputs = generate_inputs_from_targets(noisy_dct, self.transform_config)[0]
        targets = clean_dct[0]
        return inputs[:, None, :, :].astype(np.float32), targets.astype(np.float32)


@torch.no_grad()
def denoise_deepseg(
    signal: np.ndarray,
    model: DeepSeg,
    transform_config: TransformConfig,
) -> np.ndarray:
    device = next(model.parameters()).device
    windows = stdct_deepseg(signal, transform_config)
    mean = np.mean(windows)
    scale = max(float(np.std(windows)), 1e-8)
    windows = (windows - mean) / scale
    inputs = generate_inputs_from_targets(windows, transform_config)[0]
    tensor = torch.from_numpy(inputs.astype(np.float32))
    tensor = tensor[:, None, :, :].to(device)
    model.eval()
    predictions = model(tensor).cpu().numpy() * scale
    return ist_dct_deepseg(predictions[None, ...], transform_config)[0].astype(
        np.float32
    )
