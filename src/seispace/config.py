from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _filtered_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


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
        # Keep the original integer-overlap arithmetic:
        # overlap = int(patch_length * ratio), stride = patch_length - overlap.
        overlap = int(self.patch_length * self.overlap_ratio)
        return self.patch_length - overlap

    @property
    def pad_segments(self) -> int:
        return self.num_segments // 2

    @property
    def pad_length(self) -> int:
        return self.stride * self.pad_segments


@dataclass
class ModelConfig:
    name: str = "dual_sparse_prior_1d"
    embed_dim: int = 512
    base_channels: list[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 128, 64, 32, 16]
    )
    embed_loss_weight: float = 0.0


@dataclass
class DataConfig:
    kind: str = "wavelet_mat"
    root: str = ""
    train_file: str = "training_data.mat"
    test_file: str = "testing_data.mat"
    train_hdf5: str = "high_60dB_HH.hdf5"
    train_meta: str = "high_60dB_HH.csv"
    noise_hdf5: str = "noise_HH.hdf5"
    test_hdf5: str = "low_HH.hdf5"
    test_meta: str = "low_HH.csv"
    data_length: int = 2000
    snr_threshold_db: float = 70.0
    clean_mode: str = "p_and_s"
    normalization: str = "std"
    window_normalization: str = "std"
    inference_normalization: str = "window"
    max_train_traces: int | None = None


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-2
    weight_decay: float = 0.0
    scheduler_step: int = 1
    scheduler_gamma: float = 0.85
    loss: str = "mae"
    snr_db: list[float] = field(default_factory=lambda: [-20.0, 20.0])
    num_workers: int = 0
    amp: bool = True
    seed: int = 42
    log_interval: int = 20


@dataclass
class EvalConfig:
    snr_db: list[float] = field(
        default_factory=lambda: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    )
    max_signals: int | None = None
    max_traces: int | None = None
    trace_start: int = 0


@dataclass
class ExperimentConfig:
    experiment: str = "wavelet"
    output_dir: str = "runs/experiment"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        top = _filtered_kwargs(cls, values)
        top["data"] = DataConfig(**_filtered_kwargs(DataConfig, values.get("data", {})))
        top["transform"] = TransformConfig(
            **_filtered_kwargs(TransformConfig, values.get("transform", {}))
        )
        top["model"] = ModelConfig(**_filtered_kwargs(ModelConfig, values.get("model", {})))
        top["train"] = TrainConfig(**_filtered_kwargs(TrainConfig, values.get("train", {})))
        top["eval"] = EvalConfig(**_filtered_kwargs(EvalConfig, values.get("eval", {})))
        return cls(**top)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def resolve_paths(self, data_root: str | None = None, output_dir: str | None = None) -> None:
        if data_root:
            self.data.root = str(Path(data_root))
        if output_dir:
            self.output_dir = str(Path(output_dir))


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    values = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig.from_dict(values)
