import numpy as np

from seispace.classical import fx_decon_2d, wavelet_denoise_1d


def test_wavelet_denoise_shape():
    signal = np.random.default_rng(0).normal(size=2048).astype(np.float32)
    output = wavelet_denoise_1d(signal)
    assert output.shape == signal.shape
    assert np.isfinite(output).all()


def test_fx_decon_shape():
    section = np.random.default_rng(0).normal(size=(256, 64)).astype(np.float32)
    output = fx_decon_2d(section)
    assert output.shape == section.shape
    assert np.isfinite(output).all()
