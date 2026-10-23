import pytest

torch = pytest.importorskip("torch")

from seispace.baselines.msmha import MSMHA


def test_msmha_output_shapes():
    model = MSMHA(
        patch_height=16,
        patch_width=16,
        first_dim=16,
        second_dim=64,
        num_heads=4,
    )
    inputs = torch.randn(2, 16, 16)
    first, second = model(inputs)
    assert first.shape == inputs.shape
    assert second.shape == inputs.shape
