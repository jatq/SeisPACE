from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.calibrator1d import SeisPACECalibrator1D  # noqa: E402
from seispace.config import load_config  # noqa: E402
from seispace.engine import load_dual_sparse_prior  # noqa: E402
from seispace.prior2d import DualSparsePrior2D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SeisPACE inference")
    parser.add_argument("mode", choices=["1d", "2d"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-config", type=Path, required=True)
    parser.add_argument("--prior-checkpoint", type=Path, required=True)
    parser.add_argument("--calibrator-config", type=Path, required=True)
    parser.add_argument("--calibrator-checkpoint", type=Path, required=True)
    parser.add_argument("--scrn-root", type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prior2d(config_path: Path, checkpoint: Path, device: torch.device):
    config = json.loads(config_path.read_text(encoding="utf-8"))["model"]
    model = DualSparsePrior2D(
        patch_height=int(config["patch_height"]),
        patch_width=int(config["patch_width"]),
        stride_height=int(config["stride_height"]),
        stride_width=int(config["stride_width"]),
        embed_dim=int(config["embed_dim"]),
        dae_channels=tuple(config["dae_channels"]),
        init_embedding=bool(config["init_embedding"]),
        embed_loss_weight=float(config["embed_loss_weight"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("state_dict", state))
    return model.to(device).eval()


def infer_1d(args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    noisy = np.asarray(np.load(args.input), dtype=np.float32).squeeze()
    if noisy.ndim != 1:
        raise ValueError("1D inference expects a vector or a one-column array")
    prior_model = load_dual_sparse_prior(
        args.prior_checkpoint,
        load_config(args.prior_config),
        device,
    )
    prior = prior_model.predict_signal(noisy)
    config = json.loads(args.calibrator_config.read_text(encoding="utf-8"))
    calibrator = SeisPACECalibrator1D(
        channels=tuple(config["model"]["channels"]),
        in_channels=2,
        residual=False,
        zero_head=False,
    ).to(device)
    state = torch.load(args.calibrator_checkpoint, map_location="cpu", weights_only=False)
    calibrator.load_state_dict(state.get("state_dict", state))
    calibrator.eval()
    inputs = torch.from_numpy(
        np.stack([noisy, prior]).astype(np.float32)
    )[None].to(device)
    with torch.no_grad():
        estimate, _ = calibrator(inputs)
    return {"noisy": noisy, "prior": prior, "estimate": estimate[0, 0].cpu().numpy()}


def infer_2d(args: argparse.Namespace, device: torch.device) -> dict[str, np.ndarray]:
    if args.scrn_root is None:
        raise ValueError("--scrn-root is required for 2D inference")
    noisy = np.asarray(np.load(args.input), dtype=np.float32)
    if noisy.ndim != 2:
        raise ValueError("2D inference expects a section array")
    noisy_tensor = torch.from_numpy(noisy)[None, None].to(device)
    prior_model = build_prior2d(args.prior_config, args.prior_checkpoint, device)
    with torch.no_grad():
        prior = prior_model.denoise(noisy_tensor)
    scrn = load_module(
        "seispace_inference_scrn",
        args.scrn_root.resolve() / "model" / "SCRN.py",
    ).SCRN(in_nc=2)
    scrn.m_tail[0] = nn.Conv2d(scrn.dim, 1, 3, 1, 1, bias=False)
    state = torch.load(args.calibrator_checkpoint, map_location="cpu", weights_only=False)
    scrn.load_state_dict(state.get("state_dict", state))
    scrn = scrn.to(device).eval()
    with torch.no_grad():
        estimate = scrn(torch.cat([noisy_tensor, prior], dim=1))
    return {
        "noisy": noisy,
        "prior": prior[0, 0].cpu().numpy(),
        "estimate": estimate[0, 0].cpu().numpy(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result = infer_1d(args, device) if args.mode == "1d" else infer_2d(args, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
