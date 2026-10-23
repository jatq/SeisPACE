from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.utils.data as tdata
from scipy.io import loadmat

from .config import TransformConfig
from .transforms import encode_pair, mix_at_snr, normalize_std


def _read_pair_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(str(path))
    if "R" not in data or "z" not in data:
        raise KeyError(f"{path} must contain variables 'R' and 'z'")
    signals = normalize_std(data["R"].T.astype(np.float32))[0]
    noises = normalize_std(data["z"].T.astype(np.float32))[0]
    return signals, noises


class WaveletMatDataset(tdata.Dataset):
    def __init__(
        self,
        path: str | Path,
        transform_config: TransformConfig,
        training: bool = True,
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-20.0, 20.0),
        normalization: str = "std",
        window_normalization: str = "std",
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.signals, self.noises = _read_pair_file(path)
        self.transform_config = transform_config
        self.training = training
        self.snr_db = snr_db
        self.snr_range = snr_range
        self.normalization = normalization
        self.window_normalization = window_normalization
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
        noise = self.noises[noise_index]
        snr = (
            float(self.rng.uniform(*self.snr_range))
            if self.snr_db is None
            else float(self.snr_db)
        )
        clean, noisy = mix_at_snr(clean, noise, snr)
        return encode_pair(
            clean,
            noisy,
            self.transform_config,
            normalization=self.normalization,
            window_normalization=self.window_normalization,
        )


def _parse_float_vector(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = value.strip().strip("[]")
        return np.fromstring(value, sep=" ", dtype=np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _parse_first_float(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip().strip("[]")
        return float(np.fromstring(value, sep=" ", dtype=np.float32).reshape(-1)[0])
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _read_hdf5_dataset(
    path: str | Path, trace_names: list[str] | None = None
) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for STEAD HDF5 files") from exc

    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"{path} does not contain a 'data' group")
        names = (
            list(handle["data"].keys())
            if trace_names is None
            else [str(name) for name in trace_names if str(name) in handle["data"]]
        )
        return {name: np.asarray(handle["data"][name]) for name in names}


class SteadPairDataset(tdata.Dataset):
    def __init__(
        self,
        root: str | Path,
        transform_config: TransformConfig,
        signal_hdf5: str = "high_60dB_HH.hdf5",
        signal_meta: str = "high_60dB_HH.csv",
        noise_hdf5: str = "noise_HH.hdf5",
        data_length: int = 2000,
        snr_threshold_db: float = 70.0,
        clean_mode: str = "p_and_s",
        snr_range: tuple[float, float] = (-10.0, 10.0),
        normalization: str = "std",
        window_normalization: str = "std",
        max_traces: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        root = Path(root)
        self.transform_config = transform_config
        self.normalization = normalization
        self.window_normalization = window_normalization
        self.snr_range = snr_range
        self.rng = np.random.default_rng(seed)

        meta = pd.read_csv(root / signal_meta, sep="\t", low_memory=False)
        if max_traces is not None:
            meta = meta.iloc[:max_traces].reset_index(drop=True)
        signals = _read_hdf5_dataset(root / signal_hdf5, meta.trace_name.tolist())
        noises = _read_hdf5_dataset(root / noise_hdf5)

        self.clean = self._build_clean_segments(
            signals,
            meta,
            data_length=data_length,
            snr_threshold_db=snr_threshold_db,
            clean_mode=clean_mode,
        )
        noise_array = np.stack(list(noises.values()), axis=0).astype(np.float32)
        noise_array = noise_array.reshape(-1, noise_array.shape[-1])[:, :data_length]
        noise_array = self._filter_zero_traces(noise_array)
        self.noises = normalize_std(noise_array)[0]

    @staticmethod
    def _filter_zero_traces(array: np.ndarray) -> np.ndarray:
        keep = np.std(array, axis=1) > 1e-8
        return array[keep]

    def _build_clean_segments(
        self,
        signals: dict[str, np.ndarray],
        meta: pd.DataFrame,
        data_length: int,
        snr_threshold_db: float,
        clean_mode: str,
    ) -> np.ndarray:
        segments: list[np.ndarray] = []
        for row in meta.itertuples(index=False):
            trace = signals.get(str(row.trace_name))
            if trace is None:
                continue
            trace = np.asarray(trace, dtype=np.float32)
            if trace.ndim == 1:
                trace = trace[None, :]
            p = int(float(row.p_arrival_sample))
            s = int(float(row.s_arrival_sample))
            coda = int(_parse_first_float(row.coda_end_sample))
            snr = _parse_float_vector(row.snr_db)
            selected = np.where(snr > snr_threshold_db)[0]
            if selected.size == 0:
                continue

            for channel in selected:
                if channel >= trace.shape[0]:
                    continue
                if clean_mode == "p_and_s":
                    windows = [
                        trace[channel, 0 : min(s, data_length)],
                        trace[channel, s : min(s + data_length, coda)],
                    ]
                elif clean_mode == "coda":
                    windows = [trace[channel, 0 : min(coda, data_length)]]
                elif clean_mode == "p_window":
                    start = max(0, p - 50)
                    windows = [trace[channel, start : min(start + data_length, coda)]]
                else:
                    raise ValueError(f"Unknown clean_mode: {clean_mode}")

                for window in windows:
                    if window.size == 0:
                        continue
                    pad = data_length - window.size
                    window = np.pad(
                        window,
                        (pad // 2, pad // 2 + pad % 2),
                        mode="constant",
                    )
                    segments.append(window[:data_length])
        if not segments:
            raise RuntimeError("No valid STEAD clean segments were found")
        return normalize_std(np.stack(segments, axis=0).astype(np.float32))[0]

    def __len__(self) -> int:
        return self.clean.shape[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        clean = self.clean[index]
        noise = self.noises[int(self.rng.integers(self.noises.shape[0]))]
        snr = float(self.rng.uniform(*self.snr_range))
        clean, noisy = mix_at_snr(clean, noise, snr)
        return encode_pair(
            clean,
            noisy,
            self.transform_config,
            normalization=self.normalization,
            window_normalization=self.window_normalization,
        )


class SteadTraceDataset(tdata.Dataset):
    def __init__(
        self,
        root: str | Path,
        trace_hdf5: str = "low_HH.hdf5",
        trace_meta: str = "low_HH.csv",
        max_traces: int | None = None,
        trace_start: int = 0,
    ) -> None:
        super().__init__()
        root = Path(root)
        meta = pd.read_csv(root / trace_meta, sep="\t", low_memory=False)
        meta = meta.iloc[trace_start:].reset_index(drop=True)
        if max_traces is not None:
            meta = meta.iloc[:max_traces].reset_index(drop=True)
        self.meta = meta
        self.signals = _read_hdf5_dataset(root / trace_hdf5, meta.trace_name.tolist())

    def __len__(self) -> int:
        return self.meta.shape[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.meta.iloc[index]
        trace = np.asarray(self.signals[str(row.trace_name)], dtype=np.float32)
        if trace.ndim == 1:
            trace = trace[None, :]
        return {
            "signal": trace[0],
            "p_sample": int(float(row.p_arrival_sample)),
            "s_sample": int(float(row.s_arrival_sample)),
            "coda_sample": int(_parse_first_float(row.coda_end_sample)),
            "trace_name": str(row.trace_name),
        }
