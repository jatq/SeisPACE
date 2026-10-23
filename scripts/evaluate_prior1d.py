from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seispace.baselines import (  # noqa: E402
    DeepDenoiser,
    DeepSeg,
    denoise_deepdenoiser,
    denoise_deepseg,
)
from seispace.config import load_config  # noqa: E402
from seispace.datasets import SteadTraceDataset  # noqa: E402
from seispace.engine import (  # noqa: E402
    evaluate_stead_traces,
    evaluate_wavelet,
    load_dual_sparse_prior,
    resolve_device,
    save_stead_results,
    save_wavelet_results,
)
from seispace.prior1d import DualSparsePrior1D  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--model",
        choices=["dual_sparse_prior_1d", "deepdenoiser", "deepseg"],
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--inference-normalization", choices=["window", "trace"], default=None
    )
    parser.add_argument("--max-signals", type=int)
    parser.add_argument("--max-traces", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.resolve_paths(
        data_root=str(args.data_root) if args.data_root else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
    )
    model_name = args.model or config.model.name
    if args.device:
        config.device = args.device
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "dual_sparse_prior_1d":
        model = load_dual_sparse_prior(args.checkpoint, config, device)
        if args.inference_normalization:
            model.inference_normalization = args.inference_normalization
        if config.data.kind == "wavelet_mat":
            results = evaluate_wavelet(
                model.predict_signal,
                Path(config.data.root) / config.data.test_file,
                config.eval.snr_db,
                max_signals=args.max_signals or config.eval.max_signals,
            )
            save_wavelet_results(results, output_dir / "wavelet_results.csv", model_name)
            (output_dir / "wavelet_results.json").write_text(
                json.dumps(
                    {key: float(np.mean(values)) for key, values in results.items()},
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif config.data.kind == "stead_hdf5":
            dataset = SteadTraceDataset(
                config.data.root,
                trace_hdf5=config.data.test_hdf5,
                trace_meta=config.data.test_meta,
                max_traces=args.max_traces or config.eval.max_traces,
                trace_start=config.eval.trace_start,
            )
            rows = evaluate_stead_traces(model, dataset, max_traces=args.max_traces)
            save_stead_results(rows, output_dir / "stead_results.json")
            gains = [float(row["snr_gain_db"]) for row in rows if np.isfinite(row["snr_gain_db"])]
            summary = {
                "n": len(gains),
                "mean_gain_db": float(np.mean(gains)) if gains else float("nan"),
                "median_gain_db": float(np.median(gains)) if gains else float("nan"),
            }
            (output_dir / "stead_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(json.dumps(summary, indent=2))
        else:
            raise ValueError(f"Unknown data kind: {config.data.kind}")
        return

    if config.data.kind != "wavelet_mat":
        raise ValueError("Baseline evaluation is currently implemented for wavelet_mat only")
    if model_name == "deepdenoiser":
        model = DeepDenoiser(input_shape=(31, 138)).to(device)
        predict_fn = lambda signal: denoise_deepdenoiser(signal, model)
    else:
        model = DeepSeg().to(device)
        predict_fn = lambda signal: denoise_deepseg(signal, model, config.transform)
    payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(payload.get("state_dict", payload))
    model.eval()

    results = evaluate_wavelet(
        predict_fn,
        Path(config.data.root) / config.data.test_file,
        config.eval.snr_db,
        max_signals=args.max_signals or config.eval.max_signals,
    )
    save_wavelet_results(results, output_dir / f"{model_name}_wavelet_results.csv", model_name)


if __name__ == "__main__":
    main()
