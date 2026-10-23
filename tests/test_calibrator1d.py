import pytest

torch = pytest.importorskip("torch")

from seispace.calibrator1d import SeisPACECalibrator1D


def test_refiner1d_starts_as_identity_on_prior():
    model = SeisPACECalibrator1D(channels=(8, 16, 32, 64))
    inputs = torch.randn(2, 2, 2048)
    estimate, correction = model(inputs)
    assert estimate.shape == (2, 1, 2048)
    assert correction.shape == (2, 1, 2048)
    assert torch.allclose(estimate, inputs[:, 1:2], atol=1e-6)
