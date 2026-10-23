import numpy as np

from seispace.prior_control import apply_prior_control


def test_zero_prior_control_preserves_shape():
    prior = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    controlled = apply_prior_control(prior, "zero")
    assert controlled.shape == prior.shape
    assert np.count_nonzero(controlled) == 0


def test_shuffled_prior_control_is_a_derangement():
    prior = np.arange(20, dtype=np.float32).reshape(10, 2)
    controlled = apply_prior_control(prior, "shuffled", seed=7)
    assert controlled.shape == prior.shape
    assert not np.any(np.all(controlled == prior, axis=1))
