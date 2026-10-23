import pytest

torch = pytest.importorskip("torch")

from seispace.prior2d import DualSparsePrior2D


def test_prior2d_preserves_shape():
    model = DualSparsePrior2D(
        patch_height=8,
        patch_width=8,
        stride_height=4,
        stride_width=4,
        embed_dim=64,
        dae_channels=(32, 16),
    )
    inputs = torch.randn(2, 1, 64, 64)
    assert model(inputs).shape == inputs.shape
    assert model.denoise(inputs).shape == inputs.shape
    reconstructed = model.idct(model.dct(inputs))
    assert torch.allclose(reconstructed, inputs, atol=1e-5, rtol=1e-4)
