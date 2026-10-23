from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.utils.data as tdata
from scipy.io import loadmat

from .transforms import normalize_std


class WaveletSectionDataset(tdata.Dataset):
    """Build 2D sections from the paper's 1D wavelet and colored-noise data."""

    def __init__(
        self,
        mat_path: str | Path,
        num_samples: int,
        shape: tuple[int, int] = (256, 128),
        training: bool = True,
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-10.0, 10.0),
        seed: int = 42,
    ) -> None:
        super().__init__()
        data = loadmat(str(mat_path))
        self.signals = normalize_std(data["R"].T.astype(np.float32))[0]
        self.noises = normalize_std(data["z"].T.astype(np.float32))[0]
        self.num_samples = int(num_samples)
        self.shape = tuple(shape)
        self.training = training
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def _random_waveform(self, rng: np.random.Generator) -> np.ndarray:
        waveform = self.signals[int(rng.integers(self.signals.shape[0]))]
        time_samples = self.shape[0]
        center = int(np.argmax(np.abs(waveform)))
        start = int(np.clip(center - time_samples // 2, 0, waveform.size - time_samples))
        return waveform[start : start + time_samples]

    def _clean_section(self, rng: np.random.Generator) -> np.ndarray:
        time_samples, traces = self.shape
        section = np.zeros(self.shape, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            waveform = self._random_waveform(rng)
            slope = float(rng.uniform(-0.8, 0.8))
            curvature = float(rng.uniform(-0.002, 0.002))
            amplitude = float(rng.uniform(0.5, 1.5))
            for trace in range(traces):
                shift = int(round(slope * trace + curvature * trace * trace))
                section[:, trace] += amplitude * np.roll(waveform, shift)
        return normalize_std(section)[0]

    def _noise_section(
        self, rng: np.random.Generator, target_std: float
    ) -> np.ndarray:
        time_samples, traces = self.shape
        section = np.zeros(self.shape, dtype=np.float32)
        for trace in range(traces):
            noise = self.noises[int(rng.integers(self.noises.shape[0]))]
            start = int(rng.integers(0, noise.size - time_samples + 1))
            section[:, trace] = noise[start : start + time_samples]
        section = normalize_std(section)[0]
        return section * target_std

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 104729 * index)
        clean = self._clean_section(rng)
        if self.snr_db is None:
            snr = float(rng.uniform(*self.snr_range))
        else:
            snr = float(self.snr_db)
        noise_std = 10.0 ** (-snr / 20.0)
        noisy = clean + self._noise_section(rng, noise_std)
        return clean.astype(np.float32), noisy.astype(np.float32)


class CachedSectionDataset(tdata.Dataset):
    def __init__(self, root: str | Path, prefix: str) -> None:
        root = Path(root)
        self.clean = np.load(root / f"{prefix}_clean.npy")
        self.noisy = np.load(root / f"{prefix}_noisy.npy")

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.clean[index].astype(np.float32),
            self.noisy[index].astype(np.float32),
        )
