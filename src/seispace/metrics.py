from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def snr_db(clean: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    clean = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    length = min(clean.size, estimate.size)
    clean = clean[:length]
    estimate = estimate[:length]
    signal_power = np.var(clean)
    noise_power = np.var(clean - estimate)
    return float(10.0 * np.log10((signal_power + eps) / (noise_power + eps)))


def snr_gain(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> float:
    return snr_db(clean, denoised) - snr_db(clean, noisy)


def structural_similarity(
    reference: np.ndarray,
    estimate: np.ndarray,
    sigma: float = 1.5,
    data_range: float | None = None,
) -> float:
    """Local Gaussian-window SSIM for two-dimensional sections."""
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
    data_range = max(float(data_range), 1e-12)

    mu_reference = gaussian_filter(reference, sigma=sigma, mode="reflect")
    mu_estimate = gaussian_filter(estimate, sigma=sigma, mode="reflect")
    var_reference = (
        gaussian_filter(reference * reference, sigma=sigma, mode="reflect")
        - mu_reference * mu_reference
    )
    var_estimate = (
        gaussian_filter(estimate * estimate, sigma=sigma, mode="reflect")
        - mu_estimate * mu_estimate
    )
    covariance = (
        gaussian_filter(reference * estimate, sigma=sigma, mode="reflect")
        - mu_reference * mu_estimate
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_reference * mu_estimate + c1) * (2.0 * covariance + c2)
    denominator = (
        (mu_reference**2 + mu_estimate**2 + c1)
        * (var_reference + var_estimate + c2)
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def correlation(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-12) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    denominator = np.sqrt(np.sum(reference**2) * np.sum(estimate**2)) + eps
    return float(np.sum(reference * estimate) / denominator)


def optimal_scale(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    denominator = float(np.sum(estimate**2))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(reference * estimate) / denominator)


def energy_snr(trace: np.ndarray, p_sample: int, coda_sample: int, eps: float = 1e-12) -> float:
    """SNR estimate used in the legacy real-data notebook.

    The signal window spans P to coda. The noise window combines the pre-event
    and post-coda samples. This is not a true clean-reference SNR.
    """
    trace = np.asarray(trace, dtype=np.float64)
    p_sample = int(np.clip(p_sample, 1, trace.size - 1))
    coda_sample = int(np.clip(coda_sample, p_sample + 1, trace.size))
    signal = trace[p_sample:coda_sample]
    noise_parts = [trace[:p_sample]]
    if coda_sample < trace.size:
        noise_parts.append(trace[coda_sample:])
    noise = np.concatenate(noise_parts)
    if signal.size == 0 or noise.size == 0:
        return float("nan")
    return float(10.0 * np.log10((np.var(signal) + eps) / (np.var(noise) + eps)))
