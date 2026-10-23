import numpy as np

from seispace.config import TransformConfig
from seispace.transforms import ist_dct, st_dct


def test_legacy_stride():
    cfg = TransformConfig()
    assert cfg.stride == 13
    assert cfg.pad_length == 91


def test_dct_round_trip():
    rng = np.random.default_rng(0)
    cfg = TransformConfig()
    for length in [300, 2048, 6000]:
        signal = rng.normal(size=length).astype(np.float32)
        recovered = ist_dct(st_dct(signal, cfg), cfg)[0]
        usable = (
            ((length - cfg.patch_length) // cfg.stride) * cfg.stride
            + cfg.patch_length
            if length >= cfg.patch_length
            else cfg.patch_length
        )
        central = recovered[cfg.pad_length : cfg.pad_length + usable]
        assert np.max(np.abs(signal[:usable] - central)) < 1e-4
