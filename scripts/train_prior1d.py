from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.utils.data as tdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.baselines import (  # noqa: E402
    DeepDenoiser,
    DeepDenoiserWaveletDataset,
    DeepSeg,
    DeepSegWaveletDataset,
)
from seispace.config import load_config  # noqa: E402
from seispace.datasets import SteadPairDataset, WaveletMatDataset  # noqa: E402
from seispace.engine import resolve_device, set_seed, train_model  # noqa: E402
from seispace.prior1d import DualSparsePrior1D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the dual-sparse prior or a seismic denoising baseline"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model",
        choices=["dual_sparse_prior_1d", "deepdenoiser", "deepseg"],
    )
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-traces", type=int)
    return parser.parse_args()


def build_dataset(config, model_name: str):
    data = config.data
    transform = config.transform
    common = {
        "training": True,
        "snr_range": tuple(config.train.snr_db),
        "seed": config.train.seed,
    }
    if data.kind == "wavelet_mat":
        path = Path(data.root) / data.train_file
        if model_name == "dual_sparse_prior_1d":
            return WaveletMatDataset(
                path,
                transform,
                normalization=data.normalization,
                window_normalization=data.window_normalization,
                **common,
            )
        if model_name == "deepdenoiser":
            return DeepDenoiserWaveletDataset(path, **common)
        return DeepSegWaveletDataset(path, transform, **common)

    if data.kind == "stead_hdf5":
        if model_name != "dual_sparse_prior_1d":
            raise ValueError(
                "STEAD training is currently implemented for the dual-sparse prior only"
            )
        return SteadPairDataset(
            data.root,
            transform,
            signal_hdf5=data.train_hdf5,
            signal_meta=data.train_meta,
            noise_hdf5=data.noise_hdf5,
            data_length=data.data_length,
            snr_threshold_db=data.snr_threshold_db,
            clean_mode=data.clean_mode,
            normalization=data.normalization,
            window_normalization=data.window_normalization,
            max_traces=data.max_train_traces,
            seed=config.train.seed,
            snr_range=tuple(config.train.snr_db),
        )
    raise ValueError(f"Unknown data kind: {data.kind}")


def build_model(config, model_name: str):
    if model_name == "dual_sparse_prior_1d":
        return DualSparsePrior1D(
            embed_dim=config.model.embed_dim,
            transform_config=config.transform,
            base_channels=config.model.base_channels,
            inference_normalization=config.data.inference_normalization,
            embed_loss_weight=config.model.embed_loss_weight,
        )
    if model_name == "deepdenoiser":
        return DeepDenoiser(input_shape=(31, 138))
    return DeepSeg()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.resolve_paths(
        data_root=str(args.data_root) if args.data_root else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )
    model_name = args.model or config.model.name
    config.model.name = model_name
    if args.device:
        config.device = args.device
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.max_train_traces is not None:
        config.data.max_train_traces = args.max_train_traces

    set_seed(config.train.seed)
    device = resolve_device(config.device)
    dataset = build_dataset(config, model_name)
    loader = tdata.DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(config, model_name).to(device)
    print(f"model={model_name} samples={len(dataset)} device={device}")

    loss_fn = None
    if model_name == "deepdenoiser":
        loss_fn = lambda module, inputs, targets: module.trainloss(
            module(inputs), targets
        )
    elif model_name == "deepseg":
        loss_fn = lambda module, inputs, targets: module.trainloss(
            module(inputs), targets, loss_type=config.train.loss
        )
    train_model(
        model,
        loader,
        config,
        device,
        config.output_dir,
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    main()
