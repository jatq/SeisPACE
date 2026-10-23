from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import segyio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--traces", type=int, default=8)
    args = parser.parse_args()
    with segyio.open(str(args.path), "r", ignore_geometry=True) as handle:
        print("traces", handle.tracecount)
        print("samples", len(handle.samples))
        print("dt_us", segyio.tools.dt(handle))
        print(
            "first_traces",
            np.asarray([handle.trace[index][:5] for index in range(args.traces)]).tolist(),
        )
        for field_name, field in (
            ("source_x", segyio.TraceField.SourceX),
            ("group_x", segyio.TraceField.GroupX),
            ("cdp_x", segyio.TraceField.CDP_X),
            ("offset", segyio.TraceField.offset),
        ):
            values = handle.attributes(field)[:]
            print(field_name, int(np.min(values)), int(np.max(values)), int(np.unique(values).size))


if __name__ == "__main__":
    main()
