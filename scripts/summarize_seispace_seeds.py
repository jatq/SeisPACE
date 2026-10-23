from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = ("m10", "m8", "m6", "m4", "m2", "0", "2", "4", "6", "8", "10")
RUN_ROOT = ROOT / "runs" / "seispace_ablation"


def series_mean(directory: Path) -> float:
    values = []
    for tag in TAGS:
        payload = json.loads(
            (directory / f"snr_{tag}.json").read_text(encoding="utf-8")
        )
        values.append(float(payload["metrics"]["gain_db"]))
    return sum(values) / len(values)


def main() -> None:
    seeds = {
        "42": {
            "baseline": ROOT / "runs" / "eval_seispace1d_baseline",
            "seispace": RUN_ROOT / "eval_1d_late_anchor05",
        },
        "43": {
            "baseline": RUN_ROOT / "seeds" / "1d_seed43" / "eval_baseline",
            "seispace": RUN_ROOT
            / "seeds"
            / "1d_seed43"
            / "eval_late_main",
        },
        "44": {
            "baseline": RUN_ROOT / "seeds" / "1d_seed44" / "eval_baseline",
            "seispace": RUN_ROOT
            / "seeds"
            / "1d_seed44"
            / "eval_late_main",
        },
    }
    rows = []
    for seed, paths in seeds.items():
        baseline = series_mean(paths["baseline"])
        seispace = series_mean(paths["seispace"])
        rows.append(
            {
                "seed": seed,
                "baseline_mean_gain_db": baseline,
                "seispace_mean_gain_db": seispace,
                "gain_over_baseline_db": seispace - baseline,
            }
        )
    summary = {
        "rows": rows,
        "baseline_mean_db": statistics.mean(
            row["baseline_mean_gain_db"] for row in rows
        ),
        "baseline_std_db": statistics.stdev(
            row["baseline_mean_gain_db"] for row in rows
        ),
        "seispace_mean_db": statistics.mean(
            row["seispace_mean_gain_db"] for row in rows
        ),
        "seispace_std_db": statistics.stdev(
            row["seispace_mean_gain_db"] for row in rows
        ),
        "mean_gain_over_baseline_db": statistics.mean(
            row["gain_over_baseline_db"] for row in rows
        ),
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "seed_summary_1d.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (RUN_ROOT / "seed_summary_1d.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
