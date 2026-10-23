import numpy as np

from seispace.metrics import (
    correlation,
    optimal_scale,
    snr_db,
    snr_gain,
    structural_similarity,
)


def test_snr_increases_after_denoising():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=4096).astype(np.float32)
    noise = rng.normal(size=4096).astype(np.float32) * 0.3
    noisy = clean + noise
    denoised = clean + noise * 0.1
    assert snr_db(clean, denoised) > snr_db(clean, noisy)
    assert snr_gain(clean, noisy, denoised) > 0


def test_structural_metrics_recover_simple_transform():
    rng = np.random.default_rng(1)
    clean = rng.normal(size=(64, 32)).astype(np.float32)
    estimate = 0.5 * clean
    assert correlation(clean, estimate) > 0.999
    assert np.isclose(optimal_scale(clean, estimate), 2.0, rtol=1e-5)
    assert structural_similarity(clean, clean) > 0.999
