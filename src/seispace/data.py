from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import encode_pair


INPUT_COLUMNS = ("sample_id", "dimension", "shape", "clean", "noisy")
ARRAY_PRECISION = 8
csv.field_size_limit(sys.maxsize)


@dataclass
class SeismicSample:
    sample_id: str
    dimension: str
    shape: tuple[int, ...]
    noisy: np.ndarray
    clean: np.ndarray | None = None


def _parse_shape(value: str, dimension: str) -> tuple[int, ...]:
    text = str(value).strip().lower()
    if dimension == "1d":
        shape = (int(text),)
    elif dimension == "2d":
        height, width = text.split("x", maxsplit=1)
        shape = (int(height), int(width))
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")
    if any(size <= 0 for size in shape):
        raise ValueError(f"Invalid sample shape: {value}")
    return shape


def _parse_array(value: str, shape: tuple[int, ...], field: str) -> np.ndarray:
    array = np.fromstring(str(value), sep=" ", dtype=np.float32)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(
            f"Column '{field}' has {array.size} values; expected {expected} for {shape}"
        )
    return array.reshape(shape)


def _format_array(array: np.ndarray) -> str:
    values = np.asarray(array, dtype=np.float32).reshape(-1)
    return " ".join(f"{value:.{ARRAY_PRECISION}g}" for value in values)


def read_csv(path: str | Path) -> list[SeismicSample]:
    """Read the single CSV format used by both training and inference."""
    path = Path(path)
    samples: list[SeismicSample] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(INPUT_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            dimension = str(row["dimension"]).strip().lower()
            shape = _parse_shape(row["shape"], dimension)
            clean_text = str(row.get("clean", "")).strip()
            samples.append(
                SeismicSample(
                    sample_id=str(row["sample_id"]),
                    dimension=dimension,
                    shape=shape,
                    noisy=_parse_array(row["noisy"], shape, "noisy"),
                    clean=(
                        _parse_array(clean_text, shape, "clean")
                        if clean_text
                        else None
                    ),
                )
            )
    if not samples:
        raise ValueError(f"{path} contains no samples")
    dimensions = {sample.dimension for sample in samples}
    if len(dimensions) != 1:
        raise ValueError(f"{path} mixes dimensions: {sorted(dimensions)}")
    return samples


def write_csv(
    path: str | Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        _format_array(value)
                        if isinstance(value, np.ndarray)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _normalize_pair(
    clean: np.ndarray,
    noisy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(noisy))
    scale = max(float(np.std(noisy)), 1e-8)
    return (clean - mean) / scale, (noisy - mean) / scale


class CSVDataset(Dataset):
    """Minimal CSV dataset for prior or baseline/calibrator training."""

    def __init__(
        self,
        path: str | Path,
        dimension: str,
        task: str,
        patch_length: int = 256,
        overlap_ratio: float = 0.703125,
        num_segments: int = 15,
    ) -> None:
        self.samples = read_csv(path)
        if self.samples[0].dimension != dimension:
            raise ValueError(
                f"Expected {dimension} data, found {self.samples[0].dimension}"
            )
        if task not in {"prior", "pair"}:
            raise ValueError(f"Unsupported task: {task}")
        if any(sample.clean is None for sample in self.samples):
            raise ValueError(f"{path} must contain clean traces for training")
        self.dimension = dimension
        self.task = task
        self.patch_length = int(patch_length)
        self.overlap_ratio = float(overlap_ratio)
        self.num_segments = int(num_segments)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        assert sample.clean is not None
        clean, noisy = _normalize_pair(sample.clean, sample.noisy)
        if self.dimension == "1d" and self.task == "prior":
            from .config import TransformConfig

            noise, target = encode_pair(
                clean,
                noisy,
                TransformConfig(
                    patch_length=self.patch_length,
                    overlap_ratio=self.overlap_ratio,
                    num_segments=self.num_segments,
                ),
                normalization="std",
                window_normalization="std",
            )
            return (
                torch.from_numpy(noise.astype(np.float32)),
                torch.from_numpy(target.astype(np.float32)),
            )
        stacked = np.stack([noisy, clean]).astype(np.float32)
        return torch.from_numpy(stacked)
