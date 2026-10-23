from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("m10", "m8", "m6", "m4", "m2", "0", "2", "4", "6", "8", "10")
LEVELS = (-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SeisPACE experiments")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=ROOT / "configs" / "seispace_ablation_matrix.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "seispace_ablation",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(payload: dict[str, Any], metric_path: str | None) -> dict[str, Any]:
    if not metric_path:
        return payload["metrics"]
    value: Any = payload
    for key in metric_path.split("."):
        value = value[key]
    return value


def summarize_1d(root: Path, entries: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, relative in entries.items():
        found = 0
        values: list[float] = []
        for tag, level in zip(TAGS, LEVELS, strict=True):
            path = root / relative / f"snr_{tag}.json"
            if not path.exists():
                continue
            payload = read_json(path)
            gain = float(payload["metrics"]["gain_db"])
            values.append(gain)
            found += 1
            rows.append(
                {
                    "method": label,
                    "snr_db": level,
                    "gain_db": gain,
                    "correlation": float(payload["metrics"]["correlation"]),
                    "amplitude_ratio": float(
                        payload["metrics"]["amplitude_ratio"]
                    ),
                }
            )
        if found and found != len(TAGS):
            print(f"warning: {label} has only {found}/{len(TAGS)} 1D results")
        if values:
            rows.append(
                {
                    "method": label,
                    "snr_db": "mean",
                    "gain_db": sum(values) / len(values),
                    "correlation": "",
                    "amplitude_ratio": "",
                }
            )
    return rows


def summarize_2d(root: Path, entries: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, specification in entries.items():
        metric_path = specification.get("metric_path")
        for tag in ("m10", "0", "p10"):
            relative = specification.get(tag)
            if relative is None:
                continue
            path = root / relative
            if not path.exists():
                continue
            metrics = metric_value(read_json(path), metric_path)
            rows.append(
                {
                    "method": label,
                    "snr_db": tag,
                    "achieved_snr_db": float(metrics["snr_db"]),
                    "ssim": float(metrics["ssim"]),
                    "correlation": float(metrics["correlation"]),
                    "amplitude_ratio": float(metrics["amplitude_ratio"]),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No complete results yet._\n"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(
                f"{value:.4f}" if isinstance(value, float) else str(value)
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    matrix = read_json(args.matrix)
    rows_1d = summarize_1d(ROOT, matrix["one_dimensional"])
    rows_2d = summarize_2d(ROOT, matrix["two_dimensional"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_1d.csv", rows_1d)
    write_csv(args.output_dir / "summary_2d.csv", rows_2d)
    summary = {"one_dimensional": rows_1d, "two_dimensional": rows_2d}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    report = "# SeisPACE ablation summary\n\n## 1D SNR gain\n\n"
    report += markdown_table(
        rows_1d,
        ["method", "snr_db", "gain_db", "correlation", "amplitude_ratio"],
    )
    report += "\n## 2D achieved SNR\n\n"
    report += markdown_table(
        rows_2d,
        [
            "method",
            "snr_db",
            "achieved_snr_db",
            "ssim",
            "correlation",
            "amplitude_ratio",
        ],
    )
    (args.output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"wrote {len(rows_1d)} 1D rows and {len(rows_2d)} 2D rows")


if __name__ == "__main__":
    main()
