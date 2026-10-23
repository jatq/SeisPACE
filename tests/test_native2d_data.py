from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("segyio")

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_native2d_data.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_native2d_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robust_normalize_scales_whole_section():
    module = load_module()
    block = np.arange(64, dtype=np.float32).reshape(8, 8)
    result = module.robust_normalize(block)
    assert np.isclose(np.mean(result), 0.0, atol=1e-6)
    assert np.isclose(np.std(result), 1.0, atol=1e-6)
    assert not np.allclose(np.std(result, axis=-1), 1.0)


def test_robust_normalize_rejects_constant_section():
    module = load_module()
    with pytest.raises(ValueError, match="degenerate"):
        module.robust_normalize(np.ones((8, 8), dtype=np.float32))
