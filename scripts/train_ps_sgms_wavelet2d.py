from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tdata
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.synthetic2d import WaveletSectionDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain PS-SGMs on our 2D sections")
    parser.add_argument("--ps-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mat-file", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sections", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def namespace(mapping: dict) -> Namespace:
    result = Namespace()
    for key, value in mapping.items():
        setattr(result, key, namespace(value) if isinstance(value, dict) else value)
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_clean_patches(
    mat_file: Path,
    sections: int,
    patch_size: int,
    seed: int,
    cache_file: Path,
) -> np.ndarray:
    if cache_file.exists():
        patches = np.load(cache_file)
        print(f"loaded {cache_file} shape={patches.shape}", flush=True)
        return patches

    dataset = WaveletSectionDataset(
        mat_file,
        sections,
        training=True,
        snr_range=(-10.0, 10.0),
        seed=seed,
    )
    rng = np.random.default_rng(seed + 7717)
    patches = np.empty((sections, patch_size, patch_size), dtype=np.float32)
    for index in range(sections):
        clean, _ = dataset[index]
        row = int(rng.integers(0, clean.shape[0] - patch_size + 1))
        patch = clean[row : row + patch_size]
        scale = max(float(np.max(np.abs(patch))), 1e-8)
        patches[index] = patch / scale
        if (index + 1) % 200 == 0:
            print(f"prepared {index + 1}/{sections} sections", flush=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, patches)
    print(f"saved {cache_file} shape={patches.shape}", flush=True)
    return patches


def build_cached_clean_patches(
    data_dir: Path,
    sections: int,
    patch_size: int,
    seed: int,
    cache_file: Path,
) -> np.ndarray:
    if cache_file.exists():
        patches = np.load(cache_file)
        print(f"loaded {cache_file} shape={patches.shape}", flush=True)
        return patches

    clean_sections = np.load(data_dir / "train_clean.npy")
    if min(clean_sections.shape[1:]) < patch_size:
        raise ValueError(
            f"Cached sections {clean_sections.shape} are smaller than {patch_size}"
        )
    rng = np.random.default_rng(seed + 7717)
    patches = np.empty((sections, patch_size, patch_size), dtype=np.float32)
    for index in range(sections):
        section = clean_sections[index % len(clean_sections)]
        row = int(rng.integers(0, section.shape[0] - patch_size + 1))
        column = int(rng.integers(0, section.shape[1] - patch_size + 1))
        patch = np.asarray(
            section[row : row + patch_size, column : column + patch_size],
            dtype=np.float32,
        )
        scale = max(float(np.max(np.abs(patch))), 1e-8)
        patches[index] = patch / scale
        if (index + 1) % 200 == 0:
            print(f"prepared {index + 1}/{sections} cached sections", flush=True)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, patches)
    print(f"saved {cache_file} shape={patches.shape}", flush=True)
    return patches


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = namespace(raw)
    config.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config.data.channels = 1
    config.data.image_size = config.data.patch_size
    config.data.image_shape = (config.data.patch_size, config.data.patch_size)
    config.data.logit_transform = False
    config.data.rescaled = False
    config.data.seis_rescaled = True
    config.optim = Namespace(
        optimizer="Adam",
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        beta1=config.training.beta1,
        amsgrad=False,
        eps=1e-8,
    )
    if args.iterations is not None:
        config.training.iterations = args.iterations
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.sections is not None:
        config.data.sections = args.sections
    set_seed(config.training.seed)

    ps_root = args.ps_root.resolve()
    sys.path.insert(0, str(ps_root))
    from losses.dsm import anneal_dsm_score_estimation
    from models import get_sigmas
    from models.ema import EMAHelper
    from models.ncsnv2 import NCSNv2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_args = (
        args.data_dir,
        int(config.data.sections),
        int(config.data.patch_size),
        int(config.training.seed),
        args.output_dir / "clean_patches.npy",
    )
    patches = (
        build_clean_patches(args.mat_file, *patch_args[1:])
        if args.mat_file is not None
        else build_cached_clean_patches(*patch_args)
    )
    dataset = tdata.TensorDataset(torch.from_numpy(patches).unsqueeze(1))
    loader = tdata.DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=config.device.type == "cuda",
        drop_last=True,
    )
    score = NCSNv2(config).to(config.device)
    optimizer = torch.optim.Adam(
        score.parameters(),
        lr=float(config.optim.lr),
        weight_decay=float(config.optim.weight_decay),
        betas=(float(config.optim.beta1), 0.999),
        eps=float(config.optim.eps),
    )
    ema = EMAHelper(mu=float(config.model.ema_rate))
    ema.register(score)
    sigmas = get_sigmas(config)

    protocol = {
        "implementation": "mengchuangji/PS-SGMs-seismic@46f4740",
        "initialization": "random; no pretrained checkpoint loaded",
        "protocol": "unconditional score model on clean 128 x 128 wavelet-derived patches",
        "source_mat": str(args.mat_file) if args.mat_file is not None else None,
        "source_data_dir": str(args.data_dir)
        if args.data_dir is not None
        else None,
        "sections": int(config.data.sections),
        "patch_size": int(config.data.patch_size),
        "iterations": int(config.training.iterations),
        "batch_size": int(config.training.batch_size),
        "learning_rate": float(config.optim.lr),
        "sigma_begin": float(config.model.sigma_begin),
        "sigma_end": float(config.model.sigma_end),
        "num_classes": int(config.model.num_classes),
        "ngf": int(config.model.ngf),
        "seed": int(config.training.seed),
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)
    print(
        f"trainable parameters={sum(p.numel() for p in score.parameters()):,}",
        flush=True,
    )

    history_path = args.output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(
            handle, fieldnames=["iteration", "loss", "learning_rate", "seconds"]
        ).writeheader()

    iterator = iter(loader)
    for iteration in range(1, int(config.training.iterations) + 1):
        started = time.time()
        try:
            (clean,) = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            (clean,) = next(iterator)
        samples = ((clean.to(config.device, non_blocking=True) + 1.0) * 0.5).float()
        loss = anneal_dsm_score_estimation(
            score,
            samples,
            sigmas,
            labels=None,
            anneal_power=float(config.training.anneal_power),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update(score)

        value = float(loss.detach().cpu())
        row = {
            "iteration": iteration,
            "loss": value,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": float(time.time() - started),
        }
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(row)).writerow(row)
        if iteration == 1 or iteration % 100 == 0:
            print(
                f"iteration={iteration} loss={value:.6f} "
                f"seconds={row['seconds']:.3f}",
                flush=True,
            )
        if iteration % int(config.training.snapshot_freq) == 0:
            torch.save(
                {
                    "model": score.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                },
                args.output_dir / f"checkpoint_{iteration}.pt",
            )

    torch.save(
        {
            "model": score.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": int(config.training.iterations),
        },
        args.output_dir / "final.pt",
    )
    torch.save(
        {
            "model": score.state_dict(),
            "ema": ema.state_dict(),
            "iteration": int(config.training.iterations),
        },
        args.output_dir / "score.pt",
    )
    print(f"saved {args.output_dir / 'final.pt'}", flush=True)


if __name__ == "__main__":
    main()
