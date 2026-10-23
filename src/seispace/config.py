from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TransformConfig:
    patch_length: int = 128
    overlap_ratio: float = 0.9
    num_segments: int = 15
    norm: str = "ortho"

    def __post_init__(self) -> None:
        if self.patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in [0, 1)")
        if self.num_segments <= 0 or self.num_segments % 2 == 0:
            raise ValueError("num_segments must be a positive odd number")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    @property
    def stride(self) -> int:
        overlap = int(self.patch_length * self.overlap_ratio)
        return self.patch_length - overlap

    @property
    def pad_segments(self) -> int:
        return self.num_segments // 2

    @property
    def pad_length(self) -> int:
        return self.stride * self.pad_segments
