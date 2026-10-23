from pathlib import Path

import pytest

pytest.importorskip("torch")

from seispace.data import read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]


def test_example_csv_files():
    traces = read_csv(ROOT / "samples" / "1d.csv")
    sections = read_csv(ROOT / "samples" / "2d.csv")
    assert len(traces) >= 2
    assert len(sections) >= 2
    assert traces[0].dimension == "1d"
    assert sections[0].dimension == "2d"
    assert traces[0].shape == traces[0].noisy.shape
    assert sections[0].shape == sections[0].noisy.shape


def test_csv_round_trip(tmp_path: Path):
    source = read_csv(ROOT / "samples" / "1d.csv")[0]
    target = tmp_path / "roundtrip.csv"
    write_csv(
        target,
        [
            {
                "sample_id": source.sample_id,
                "dimension": source.dimension,
                "shape": source.shape[0],
                "clean": source.clean,
                "noisy": source.noisy,
            }
        ],
        ["sample_id", "dimension", "shape", "clean", "noisy"],
    )
    restored = read_csv(target)[0]
    assert restored.sample_id == source.sample_id
    assert restored.shape == source.shape
