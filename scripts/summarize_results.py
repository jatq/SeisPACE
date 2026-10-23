from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine evaluation outputs")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    return parser.parse_args()


def read_wavelet_result(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    first = frame.columns[0]
    if first != "model":
        frame = frame.rename(columns={first: "model"})
    return frame


def main() -> None:
    args = parse_args()
    root = args.runs_root
    candidates = {
        "DualSparsePrior1D": root / "eval_wavelet_seispace" / "wavelet_results.csv",
        "DualSparsePrior1D-legacy": root
        / "eval_wavelet_seispace_legacy"
        / "wavelet_results.csv",
        "DualSparsePrior1D-2023": root
        / "eval_wavelet_seispace_2023"
        / "wavelet_results.csv",
        "DualSparsePrior1D-GitHub": root
        / "eval_github_seispace"
        / "wavelet_results.csv",
        "DeepDenoiser": root
        / "eval_wavelet_deepdenoiser"
        / "deepdenoiser_wavelet_results.csv",
        "DeepSeg": root / "eval_wavelet_deepseg" / "deepseg_wavelet_results.csv",
        "Wavelet-threshold": root
        / "eval_wavelet_classical"
        / "wavelet_results.csv",
        "DeepDenoiser-2023": root
        / "repro_deepdenoiser"
        / "deepdenoiser_wavelet_results.csv",
        "DeepSeg-2023": root / "repro_deepseg2" / "deepseg_wavelet_results.csv",
    }
    rows = []
    for name, path in candidates.items():
        frame = read_wavelet_result(path)
        if frame is not None:
            frame["model"] = name
            rows.append(frame)
    if not rows:
        raise FileNotFoundError("No wavelet result files were found")

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(root / "wavelet_combined.csv", index=False)

    snr_columns = [column for column in combined.columns if column != "model"]
    markdown = ["| Model | " + " | ".join(snr_columns) + " |"]
    markdown.append("|---|" + "|".join(["---:"] * len(snr_columns)) + "|")
    for _, row in combined.iterrows():
        values = [f"{float(row[column]):.2f}" for column in snr_columns]
        markdown.append(f"| {row['model']} | " + " | ".join(values) + " |")
    (root / "wavelet_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    steady_summary = root / "eval_stead_seispace" / "stead_summary.json"
    if steady_summary.exists():
        summary = json.loads(steady_summary.read_text(encoding="utf-8"))
        (root / "stead_summary.md").write_text(
            "\n".join(
                [
                    "| Metric | Value |",
                    "|---|---:|",
                    f"| Traces | {summary['n']} |",
                    f"| Mean SNR gain (dB) | {summary['mean_gain_db']:.2f} |",
                    f"| Median SNR gain (dB) | {summary['median_gain_db']:.2f} |",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    print((root / "wavelet_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
