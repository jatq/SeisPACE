from __future__ import annotations

import numpy as np


def apply_prior_control(
    prior: np.ndarray,
    mode: str,
    seed: int = 2026,
) -> np.ndarray:
    """Create inference-time controls for a batch of prior predictions."""
    if mode == "correct":
        return prior
    if mode == "zero":
        return np.zeros_like(prior)
    if mode != "shuffled":
        raise ValueError(f"Unsupported prior control mode: {mode}")

    if prior.shape[0] < 2:
        raise ValueError("Shuffled prior control requires at least two samples")
    shift = int(np.random.default_rng(seed).integers(1, prior.shape[0]))
    return np.roll(prior, shift=shift, axis=0)
