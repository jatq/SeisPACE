from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.io import loadmat
import torch.utils.data as tdata

from ..transforms import mix_at_snr, normalize_std


class Conv3x3ReLUBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.conv(x)))


class Deconv3x3ReLUBatchNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        output_padding: tuple[int, int] = (1, 1),
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
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.conv(x)))


class DeepDenoiser(nn.Module):
    def __init__(self, input_shape: tuple[int, int] = (31, 138)) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.n_classes = 2
        self.down_depth = 5
        shortcut_channels: list[int] = []
        channels = [
            (8, 1),
            (8, 1),
            (8, 2),
            (16, 1),
            (16, 2),
            (32, 1),
            (32, 2),
            (64, 1),
            (64, 2),
            (128, 1),
            (128, 2),
        ]
        self.encoder = nn.ModuleList()
        for index, (out_channels, stride) in enumerate(channels):
            in_channels = 2 if index == 0 else channels[index - 1][0]
            self.encoder.append(Conv3x3ReLUBatchNorm(in_channels, out_channels, stride))
            if index in [1, 3, 5, 7, 9]:
                shortcut_channels.append(out_channels)

        decoder_channels = [
            (256, 1),
            (512, 2),
            (128, 1),
            (128, 2),
            (64, 1),
            (64, 2),
            (32, 1),
            (32, 2),
            (16, 1),
            (16, 2),
            (8, 1),
        ]
        output_padding = self._get_output_padding()
        self.decoder = nn.ModuleList()
        for index, (out_channels, stride) in enumerate(decoder_channels):
            shortcut = shortcut_channels.pop() if index in [2, 4, 6, 8, 10] else 0
            in_channels = (
                channels[-1][0]
                if index == 0
                else decoder_channels[index - 1][0] + shortcut
            )
            if stride == 2:
                self.decoder.append(
                    Deconv3x3ReLUBatchNorm(
                        in_channels, out_channels, output_padding.pop()
                    )
                )
            else:
                self.decoder.append(
                    Conv3x3ReLUBatchNorm(in_channels, out_channels, stride)
                )
        self.head = Conv3x3ReLUBatchNorm(decoder_channels[-1][0], self.n_classes)

    def _get_output_padding(self) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        height, width = self.input_shape
        for _ in range(self.down_depth):
            result.append((1 if height % 2 == 0 else 0, 1 if width % 2 == 0 else 0))
            height, width = (height + 1) // 2, (width + 1) // 2
        return result

    def forward(self, x: torch.Tensor, is_softmax: bool = False) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        batch, traces, _, _, _ = x.shape
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        shortcuts: list[torch.Tensor] = []
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in [1, 3, 5, 7, 9]:
                shortcuts.append(x)
        for index, layer in enumerate(self.decoder):
            if index in [2, 4, 6, 8, 10]:
                x = torch.cat([x, shortcuts.pop()], dim=1)
            x = layer(x)
        x = self.head(x)
        mask = x.reshape(batch, traces, self.n_classes, x.shape[-2], x.shape[-1])
        return nn.Softmax(dim=-3)(mask) if is_softmax else mask

    def trainloss(self, predictions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if masks.ndim == 4:
            masks = masks.unsqueeze(1)
        predictions = predictions.permute([0, 1, 3, 4, 2]).reshape(-1, self.n_classes)
        masks = masks.permute([0, 1, 3, 4, 2]).reshape(-1, self.n_classes)
        return nn.CrossEntropyLoss(reduction="mean")(predictions, masks)


def _stft(
    x: np.ndarray,
    fs: float = 100,
    nperseg: int = 30,
    nfft: int = 60,
    boundary: str = "zeros",
) -> np.ndarray:
    return signal.stft(x, fs=fs, nperseg=nperseg, nfft=nfft, boundary=boundary)[2]


def _istft(
    z: np.ndarray,
    fs: float = 100,
    nperseg: int = 30,
    nfft: int = 60,
    boundary: str = "zeros",
) -> np.ndarray:
    return signal.istft(z, fs=fs, nperseg=nperseg, nfft=nfft, boundary=boundary)[1]


class DeepDenoiserWaveletDataset(tdata.Dataset):
    def __init__(
        self,
        path: str | Path,
        training: bool = True,
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-10.0, 10.0),
        seed: int | None = None,
    ) -> None:
        super().__init__()
        data = loadmat(str(path))
        self.signals = normalize_std(data["R"].T.astype(np.float32))[0]
        self.noises = normalize_std(data["z"].T.astype(np.float32))[0]
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
        clean_ft = _stft(clean)
        noise_ft = _stft(noisy - clean)
        noisy_ft = _stft(noisy)
        noisy_ft = (noisy_ft - np.mean(noisy_ft)) / max(np.std(noisy_ft), 1e-8)
        inputs = np.stack([noisy_ft.real, noisy_ft.imag], axis=0).astype(np.float32)
        mask = np.abs(clean_ft) / (np.abs(clean_ft) + np.abs(noise_ft) + 1e-10)
        mask = np.clip(mask, 0.0, 1.0)
        targets = np.stack([mask, 1.0 - mask], axis=0).astype(np.float32)
        return inputs, targets


@torch.no_grad()
def denoise_deepdenoiser(signal: np.ndarray, model: DeepDenoiser) -> np.ndarray:
    device = next(model.parameters()).device
    signal_ft = _stft(signal)
    normalized = (signal_ft - np.mean(signal_ft)) / max(np.std(signal_ft), 1e-8)
    inputs = np.stack([normalized.real, normalized.imag], axis=0)
    tensor = torch.from_numpy(inputs.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    mask = model(tensor, is_softmax=True).cpu().numpy()[0, 0]
    predicted_ft = signal_ft * mask[0]
    predicted = _istft(predicted_ft)
    if predicted.size < signal.size:
        predicted = np.pad(predicted, (0, signal.size - predicted.size), mode="constant")
    return predicted[: signal.size].astype(np.float32)
