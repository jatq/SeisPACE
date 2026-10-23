import pytest

torch = pytest.importorskip("torch")

from seispace.prior_refiner import PriorResidualRefiner


def test_zero_correction_preserves_prior():
    backbone = torch.nn.Conv2d(2, 1, 1)
    torch.nn.init.zeros_(backbone.weight)
    torch.nn.init.zeros_(backbone.bias)
    model = PriorResidualRefiner(backbone)
    inputs = torch.randn(2, 2, 16, 16)
    estimate, correction = model(inputs)
    assert torch.allclose(estimate, inputs[:, 1:2])
    assert torch.count_nonzero(correction) == 0
