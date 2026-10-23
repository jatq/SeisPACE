from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hdf5", default="high_60dB_HH.hdf5")
    parser.add_argument("--meta", default="high_60dB_HH.csv")
    args = parser.parse_args()
    with h5py.File(args.data_root / args.hdf5, "r") as handle:
        print("root keys", list(handle.keys()))
        group = handle["data"]
        print("data type", type(group), "entries", len(group))
        names = list(group.keys())[:5]
        for name in names:
            value = np.asarray(group[name])
            print(name, value.shape, value.dtype, float(np.nanmin(value)), float(np.nanmax(value)))
    meta = pd.read_csv(args.data_root / args.meta, sep="\t", low_memory=False)
    print(meta[["trace_name", "source_id", "p_arrival_sample", "receiver_latitude", "receiver_longitude", "source_distance_km"]].to_string())
    counts = meta.groupby("source_id").size().sort_values(ascending=False)
    print("events", len(counts))
    print("event-size quantiles", counts.quantile([0.5, 0.9, 0.95, 0.99]).to_dict())
    print("events with >=128 traces", int((counts >= 128).sum()))
    print(counts.head(20).to_string())


if __name__ == "__main__":
    main()
