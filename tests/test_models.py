from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from seispace.models import build_calibrator, build_prior


ROOT = Path(__file__).resolve().parents[1]


def test_1d_models_load_bundle():
    bundle = torch.load(
        ROOT / "checkpoints" / "seispace.pt",
        map_location="cpu",
        weights_only=False,
    )
    prior = build_prior("1d", bundle["1d"]["prior"]["config"])
    prior.load_state_dict(bundle["1d"]["prior"]["state_dict"])
    calibrator = build_calibrator(
        "1d",
        2,
        bundle["1d"]["calibrator"]["config"],
    )
    calibrator.load_state_dict(bundle["1d"]["calibrator"]["state_dict"])
    assert prior.predict_signal(torch.randn(512).numpy()).shape == (512,)
    assert calibrator(torch.randn(1, 2, 512))[0].shape == (1, 1, 512)


def test_2d_calibrator_forward():
    bundle = torch.load(
        ROOT / "checkpoints" / "seispace.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = build_calibrator(
        "2d",
        2,
        bundle["2d"]["calibrator"]["config"],
    )
    model.load_state_dict(bundle["2d"]["calibrator"]["state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 2, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()
