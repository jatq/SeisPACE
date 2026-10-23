from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LEVELS = ("m10", "0", "p10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble all retrained baselines for an external SEGY source"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--no-prior-dir", required=True)
    parser.add_argument("--no-consolidation-dir", required=True)
    parser.add_argument("--seispace-dir", required=True)
    parser.add_argument("--prior-dir", required=True)
    parser.add_argument("--zero-prior-dir", default="eval_zero_prior")
    parser.add_argument("--shuffled-prior-dir", default="eval_shuffled_prior")
    parser.add_argument("--classical-dir", default="eval_classical")
    parser.add_argument("--ps-sgms-dir", default="eval_ps_sgms")
    parser.add_argument("--seisdn-dir", default="eval_seisdn")
    parser.add_argument("--msmha-dir", default="eval_msmha")
    parser.add_argument("--figure", type=Path)
    parser.add_argument("--title", default="External SEGY benchmark")
    return parser.parse_args()


def load_metrics(path: Path, key: str = "metrics") -> dict[str, float] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))[key]


def add_method(
    output: dict[str, dict[str, dict[str, float]]],
    name: str,
    paths: list[Path],
    key: str = "metrics",
) -> None:
    values = [load_metrics(path, key) for path in paths]
    if all(value is not None for value in values):
        output[name] = {
            level: value
            for level, value in zip(LEVELS, values, strict=True)
        }


def main() -> None:
    args = parse_args()
    output: dict[str, dict[str, dict[str, float]]] = {}
    method_dirs = {
        "scrn_baseline": args.baseline_dir,
        "no_prior": args.no_prior_dir,
        "no_consolidation": args.no_consolidation_dir,
        "seispace": args.seispace_dir,
        "ps_sgms": args.ps_sgms_dir,
        "seisdn": args.seisdn_dir,
        "msmha": args.msmha_dir,
        "zero_prior": args.zero_prior_dir,
        "shuffled_prior": args.shuffled_prior_dir,
    }
    for name, directory in method_dirs.items():
        add_method(
            output,
            name,
            [args.root / directory / f"{level}.json" for level in LEVELS],
        )
    add_method(
        output,
        "prior_only",
        [args.root / args.prior_dir / f"{level}.json" for level in LEVELS],
        key="prior",
    )

    classical_paths = [
        args.root / args.classical_dir / f"{level}.json" for level in LEVELS
    ]
    if all(path.exists() for path in classical_paths):
        payloads = [
            json.loads(path.read_text(encoding="utf-8"))["metrics"]
            for path in classical_paths
        ]
        for name in ("noisy", "curvelet", "fx_decon", "wavelet"):
            if all(name in payload for payload in payloads):
                output[name] = {
                    level: payload[name]
                    for level, payload in zip(LEVELS, payloads, strict=True)
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "level", "snr_db", "ssim", "correlation", "amplitude_ratio"])
        for method, levels in output.items():
            for level, metrics in levels.items():
                writer.writerow(
                    [
                        method,
                        level,
                        metrics["snr_db"],
                        metrics["ssim"],
                        metrics["correlation"],
                        metrics["amplitude_ratio"],
                    ]
                )
    print("method,m10,0,p10")
    for method, levels in output.items():
        print(
            method
            + ","
            + ",".join(f"{levels[level]['snr_db']:.3f}" for level in LEVELS)
        )
    if args.figure is not None:
        methods = [
            method
            for method in output
            if method not in {"noisy", "prior_only"}
        ]
        methods.sort(
            key=lambda method: np.mean(
                [output[method][level]["snr_db"] for level in LEVELS]
            ),
            reverse=True,
        )
        x = np.arange(len(LEVELS))
        width = 0.8 / len(methods)
        fig, axis = plt.subplots(figsize=(11.5, 4.6))
        for index, method in enumerate(methods):
            axis.bar(
                x + (index - (len(methods) - 1) / 2) * width,
                [output[method][level]["snr_db"] for level in LEVELS],
                width,
                label=method.replace("_", " "),
            )
        axis.set_xticks(x, ("-10 dB", "0 dB", "10 dB"))
        axis.set_ylabel("Achieved SNR (dB)")
        axis.set_title(args.title)
        axis.grid(axis="y", linewidth=0.4, alpha=0.3)
        axis.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "pdf"):
            fig.savefig(args.figure.with_suffix(f".{suffix}"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
