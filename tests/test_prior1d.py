import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seispace.config import TransformConfig
from seispace.prior1d import (
    DualSparsePrior1D,
    _remap_legacy_state_dict,
    _remap_upstream_state_dict,
)


def test_forward_shape():
    cfg = TransformConfig()
    model = DualSparsePrior1D(embed_dim=32, transform_config=cfg)
    inputs = torch.randn(2, 3, cfg.num_segments, cfg.patch_length)
    outputs = model(inputs)
    assert outputs.shape == (2 * 3, cfg.patch_length)


def test_predict_signal_shape():
    cfg = TransformConfig()
    model = DualSparsePrior1D(embed_dim=32, transform_config=cfg)
    signal = np.random.default_rng(0).normal(size=2048).astype(np.float32)
    output = model.predict_signal(signal)
    assert output.shape == signal.shape
    assert np.isfinite(output).all()


def test_legacy_state_dict_keys():
    remapped = _remap_legacy_state_dict(
        {
            "sparse_map.1.weight": torch.zeros(1),
            "unet.4.deconv.weight": torch.zeros(1),
            "head.0.weight": torch.zeros(1),
            "head.2.weight": torch.zeros(1),
        }
    )
    assert "sparse_mapping.1.conv.weight" in remapped
    assert "sparse_denoising.4.conv.weight" in remapped
    assert "sparse_denoising.8.conv.weight" in remapped
    assert "inverse_sparse_mapping.1.conv.weight" in remapped


def test_upstream_state_dict_keys():
    remapped = _remap_upstream_state_dict(
        {
            "dct_weight": torch.zeros(1),
            "embed.weight": torch.zeros(1),
            "inverse_embed.weight": torch.zeros(1),
            "dae.unet.0.conv.weight": torch.zeros(1),
        }
    )
    assert "dct_weight" not in remapped
    assert "sparse_mapping.1.conv.weight" in remapped
    assert "inverse_sparse_mapping.1.conv.weight" in remapped
    assert "sparse_denoising.0.conv.weight" in remapped
