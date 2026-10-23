from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("m10", "0", "p10")
RUN_ROOT = ROOT / "runs" / "seispace_ablation"


def read_metric(path: Path, metric_path: str | None = None) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    if metric_path:
        for key in metric_path.split("."):
            value = value[key]
    else:
        value = value["metrics"]
    return float(value["snr_db"])


def seed_values(seed: str) -> dict[str, dict[str, float]]:
    if seed == "42":
        baseline_dir = ROOT / "runs" / "eval_wavelet2d_with_seispace"
        main_paths = {
            "m10": RUN_ROOT / "eval_2d_late_anchor05_m10db.json",
            "0": RUN_ROOT / "eval_2d_late_anchor05_0db.json",
            "p10": RUN_ROOT / "eval_2d_late_anchor05_p10db.json",
        }
        baseline = {
            "m10": read_metric(
                baseline_dir / "summary_m10db.json", "metrics.scrn"
            ),
            "0": read_metric(baseline_dir / "summary_0db.json", "metrics.scrn"),
            "p10": read_metric(
                baseline_dir / "summary_p10db.json", "metrics.scrn"
            ),
        }
    else:
        root = RUN_ROOT / "seeds" / f"2d_seed{seed}"
        baseline = {
            level: read_metric(root / "eval_baseline" / f"{level}.json")
            for level in LEVELS
        }
        main_paths = {
            level: root / "eval_main" / f"{level}.json" for level in LEVELS
        }
    main = {level: read_metric(main_paths[level]) for level in LEVELS}
    return {"baseline": baseline, "seispace": main}


def main() -> None:
    rows = []
    per_seed = {}
    for seed in ("42", "43", "44"):
        values = seed_values(seed)
        per_seed[seed] = values
        for level in LEVELS:
            rows.append(
                {
                    "seed": seed,
                    "level": level,
                    "baseline_snr_db": values["baseline"][level],
                    "seispace_snr_db": values["seispace"][level],
                    "gain_db": values["seispace"][level]
                    - values["baseline"][level],
                }
            )
    summary = {
        "per_seed": per_seed,
        "aggregate": {},
    }
    for level in LEVELS:
        baseline = [per_seed[seed]["baseline"][level] for seed in per_seed]
        seispace = [per_seed[seed]["seispace"][level] for seed in per_seed]
        summary["aggregate"][level] = {
            "baseline_mean_db": statistics.mean(baseline),
            "baseline_std_db": statistics.stdev(baseline),
            "seispace_mean_db": statistics.mean(seispace),
            "seispace_std_db": statistics.stdev(seispace),
            "mean_gain_db": statistics.mean(
                seispace[index] - baseline[index] for index in range(len(seispace))
            ),
        }
    (RUN_ROOT / "seed_summary_2d.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (RUN_ROOT / "seed_summary_2d.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
