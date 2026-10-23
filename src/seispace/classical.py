from __future__ import annotations

import numpy as np
import pywt


def wavelet_denoise_1d(
    signal: np.ndarray,
    wavelet: str = "db8",
    level: int | None = None,
    threshold_mode: str = "soft",
) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError("wavelet_denoise_1d expects a 1D signal")
    max_level = pywt.dwt_max_level(signal.size, pywt.Wavelet(wavelet).dec_len)
    if max_level <= 0:
        return signal.copy()
    if level is None:
        level = min(4, max_level)
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode="periodization")
    finest = coeffs[-1]
    sigma = np.median(np.abs(finest - np.median(finest))) / 0.6745
    threshold = sigma * np.sqrt(2.0 * np.log(signal.size))
    coeffs = [coeffs[0]] + [
        pywt.threshold(coefficient, threshold, mode=threshold_mode)
        for coefficient in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs, wavelet, mode="periodization")
    return denoised[: signal.size].astype(np.float32)


def fx_decon_2d(
    section: np.ndarray,
    filter_length: int = 4,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Causal f-x prediction filtering.

    The section is expected to have shape (time, traces). The filter predicts
    each trace at every frequency from the preceding ``filter_length`` traces.
    """
    section = np.asarray(section, dtype=np.float32)
    if section.ndim != 2:
        raise ValueError("fx_decon_2d expects an array with shape (time, traces)")
    if section.shape[1] <= filter_length:
        return section.copy()

    mean = section.mean(axis=0, keepdims=True)
    std = section.std(axis=0, keepdims=True)
    normalized = (section - mean) / np.maximum(std, 1e-8)
    spectrum = np.fft.rfft(normalized, axis=0)
    predicted = spectrum.copy()

    for frequency in range(spectrum.shape[0]):
        trace_spectrum = spectrum[frequency]
        design = np.stack(
            [
                trace_spectrum[index - filter_length : index]
                for index in range(filter_length, trace_spectrum.size)
            ],
            axis=0,
        )
        target = trace_spectrum[filter_length:]
        gram = design.conj().T @ design
        rhs = design.conj().T @ target
        coefficients = np.linalg.solve(
            gram + ridge * np.eye(filter_length, dtype=gram.dtype),
            rhs,
        )
        for index in range(filter_length, trace_spectrum.size):
            history = trace_spectrum[index - filter_length : index]
            predicted[frequency, index] = np.dot(coefficients, history)

    denoised = np.fft.irfft(predicted, n=section.shape[0], axis=0)
    return ((denoised * std) + mean).astype(np.float32)


def _curvelet_threshold(value, threshold: float):
    if isinstance(value, list):
        return [_curvelet_threshold(item, threshold) for item in value]
    array = np.asarray(value)
    magnitude = np.abs(array)
    scale = np.maximum(magnitude - threshold, 0.0) / np.maximum(magnitude, 1e-12)
    return (array * scale).astype(np.complex64)


def _curvelet_values(value):
    if isinstance(value, list):
        for item in value:
            yield from _curvelet_values(item)
    else:
        yield np.asarray(value)


def curvelet_denoise_2d(
    section: np.ndarray,
    num_scales: int = 4,
    threshold_factor: float = 0.6,
) -> np.ndarray:
    """Uniform discrete curvelet soft-thresholding."""
    from curvelets.numpy import UDCT

    section = np.asarray(section, dtype=np.float32)
    if section.ndim != 2:
        raise ValueError("curvelet_denoise_2d expects a 2D section")
    transform = UDCT(shape=section.shape, num_scales=num_scales)
    coefficients = transform.forward(section)
    finest = list(_curvelet_values(coefficients[-1]))
    values = np.concatenate([np.abs(item).reshape(-1) for item in finest])
    sigma = np.median(np.abs(values - np.median(values))) / 0.6745
    threshold = threshold_factor * sigma * np.sqrt(
        2.0 * np.log(section.size)
    )
    thresholded = [coefficients[0]] + [
        _curvelet_threshold(item, threshold) for item in coefficients[1:]
    ]
    return np.asarray(transform.backward(thresholded), dtype=np.float32)
