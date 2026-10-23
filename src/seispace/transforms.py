from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fftpack import dct, idct

from .config import TransformConfig


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x[None, :] if x.ndim == 1 else x


def normalize_std(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, eps), std


def normalize_absmax(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    return x / np.maximum(scale, eps), scale


def pad_signal(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    x = _as_2d(x)
    if x.shape[1] < cfg.patch_length:
        x = np.pad(x, ((0, 0), (0, cfg.patch_length - x.shape[1])), mode="constant")
    usable = ((x.shape[1] - cfg.patch_length) // cfg.stride) * cfg.stride + cfg.patch_length
    x = x[:, :usable]
    return np.pad(x, ((0, 0), (cfg.pad_length, cfg.pad_length)), mode="constant")


def stack_windows(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    x = pad_signal(x, cfg)
    return sliding_window_view(x, cfg.patch_length, axis=1)[:, :: cfg.stride, :].copy()


def unstack_windows(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    windows = _as_2d(windows)
    if windows.ndim == 2:
        windows = windows[None, ...]
    channels, num_windows, patch_length = windows.shape
    if patch_length != cfg.patch_length:
        raise ValueError(f"Expected patch length {cfg.patch_length}, got {patch_length}")

    output_length = patch_length + (num_windows - 1) * cfg.stride
    output = np.zeros((channels, output_length), dtype=np.float32)
    weights = np.zeros(output_length, dtype=np.float32)
    for index, window in enumerate(windows.transpose(1, 0, 2)):
        start = index * cfg.stride
        stop = start + patch_length
        output[:, start:stop] += window
        weights[start:stop] += 1.0
    return output / np.maximum(weights[None, :], 1.0)


def st_dct(x: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    windows = stack_windows(x, cfg)
    return dct(windows, norm=cfg.norm).astype(np.float32)


def ist_dct(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    samples = idct(np.asarray(windows, dtype=np.float32), norm=cfg.norm)
    return unstack_windows(samples, cfg)


def segment_inputs(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    windows = _as_2d(windows)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (channels, num_windows, patch_length)")
    pad = cfg.pad_segments
    num_outputs = windows.shape[1] - 2 * pad
    if num_outputs <= 0:
        raise ValueError("Signal is too short for the requested number of segments")
    offset = np.arange(num_outputs)[:, None] + np.arange(2 * pad + 1)[None, :]
    return windows[:, offset, :]


def segment_targets(windows: np.ndarray, cfg: TransformConfig) -> np.ndarray:
    windows = _as_2d(windows)
    if windows.ndim != 3:
        raise ValueError("windows must have shape (channels, num_windows, patch_length)")
    pad = cfg.pad_segments
    return windows[:, pad : windows.shape[1] - pad, :]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    clean, _ = normalize_std(clean)
    noise, _ = normalize_std(noise)
    scale = float(10.0 ** (-float(snr_db) / 20.0))
    noisy = clean + scale * noise
    return clean.astype(np.float32), noisy.astype(np.float32)


def encode_pair(
    clean: np.ndarray,
    noisy: np.ndarray,
    cfg: TransformConfig,
    normalization: str = "std",
    window_normalization: str = "std",
) -> tuple[np.ndarray, np.ndarray]:
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    if normalization == "std":
        noisy_norm, scale = normalize_std(noisy)
        clean_norm = (clean - np.mean(clean)) / np.maximum(scale, 1e-8)
    elif normalization == "absmax":
        noisy_norm, scale = normalize_absmax(noisy)
        clean_norm = clean / np.maximum(scale, 1e-8)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    noisy_windows = segment_inputs(st_dct(noisy_norm, cfg), cfg)[0]
    target_windows = segment_targets(st_dct(clean_norm, cfg), cfg)[0]

    if window_normalization == "std":
        window_scale = np.std(noisy_windows, axis=(-2, -1), keepdims=True)
        window_scale = np.maximum(window_scale, 1e-8)
        noisy_windows = noisy_windows / window_scale
        target_windows = target_windows / window_scale.reshape(-1, 1)
    elif window_normalization == "absmax":
        window_scale = np.max(np.abs(noisy_windows), axis=(-2, -1), keepdims=True)
        window_scale = np.maximum(window_scale, 1e-8)
        noisy_windows = noisy_windows / window_scale
        target_windows = target_windows / window_scale.reshape(-1, 1)
    elif window_normalization != "none":
        raise ValueError(f"Unknown window normalization: {window_normalization}")

    return noisy_windows.astype(np.float32), target_windows.astype(np.float32)
