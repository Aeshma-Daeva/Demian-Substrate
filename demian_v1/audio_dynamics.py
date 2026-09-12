"""Deterministic, non-semantic acoustic dynamics features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


FEATURE_NAMES = (
    "log_rms",
    "peak",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_flatness",
    "spectral_flux",
    "low_energy_ratio",
    "mid_energy_ratio",
    "high_energy_ratio",
    "f0_normalized",
    "voicing_confidence",
    "log_rms_delta",
)


class AudioFeatureExtractor:
    """Extract frame features while retaining only temporal comparison state."""

    def __init__(self) -> None:
        self.previous_log_rms: float | None = None
        self.previous_spectrum: np.ndarray | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "previous_log_rms": self.previous_log_rms,
            "previous_spectrum": None
            if self.previous_spectrum is None
            else self.previous_spectrum.tolist(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        log_rms = snapshot.get("previous_log_rms")
        spectrum = snapshot.get("previous_spectrum")
        if log_rms is not None and (
            isinstance(log_rms, bool)
            or not isinstance(log_rms, (int, float))
            or not math.isfinite(float(log_rms))
        ):
            raise ValueError("audio_feature_snapshot_invalid")
        if spectrum is not None:
            try:
                restored = np.asarray(spectrum, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as error:
                raise ValueError("audio_feature_snapshot_invalid") from error
            if not np.all(np.isfinite(restored)):
                raise ValueError("audio_feature_snapshot_invalid")
            self.previous_spectrum = restored
        else:
            self.previous_spectrum = None
        self.previous_log_rms = None if log_rms is None else float(log_rms)

    def extract(self, frame: np.ndarray, sample_rate: int) -> dict[str, float]:
        samples = np.asarray(frame, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            raise ValueError("audio_frame_empty")
        if sample_rate <= 0 or not np.all(np.isfinite(samples)):
            raise ValueError("audio_frame_invalid")

        rms = float(np.sqrt(np.mean(samples * samples)))
        log_rms = float(np.log10(max(rms, 1e-12)))
        peak = float(np.max(np.abs(samples)))
        zero_crossing_rate = (
            float(np.mean(np.signbit(samples[1:]) != np.signbit(samples[:-1])))
            if samples.size > 1
            else 0.0
        )

        window = np.hanning(samples.size) if samples.size > 2 else np.ones(samples.size)
        magnitude = np.abs(np.fft.rfft(samples * window))
        power = magnitude * magnitude
        total_power = float(power.sum())
        frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate)
        if total_power > 1e-24:
            normalized = power / total_power
            centroid_hz = float(np.sum(frequencies * normalized))
            bandwidth_hz = float(
                np.sqrt(np.sum(((frequencies - centroid_hz) ** 2) * normalized))
            )
            flatness = float(
                np.exp(np.mean(np.log(magnitude + 1e-12)))
                / (np.mean(magnitude) + 1e-12)
            )
        else:
            normalized = np.zeros_like(power)
            centroid_hz = bandwidth_hz = flatness = 0.0

        previous = self.previous_spectrum
        if previous is None or previous.size == 0:
            spectral_flux = 0.0
        else:
            if previous.size != normalized.size:
                old_axis = np.linspace(0.0, 1.0, previous.size)
                new_axis = np.linspace(0.0, 1.0, normalized.size)
                previous = np.interp(new_axis, old_axis, previous)
            spectral_flux = float(np.sqrt(np.mean((normalized - previous) ** 2)))

        def energy_ratio(lower: float, upper: float | None) -> float:
            mask = frequencies >= lower
            if upper is not None:
                mask &= frequencies < upper
            return float(power[mask].sum() / total_power) if total_power > 1e-24 else 0.0

        f0_hz, voicing = self._estimate_f0(samples, sample_rate)
        delta = 0.0 if self.previous_log_rms is None else log_rms - self.previous_log_rms
        features = {
            "log_rms": log_rms,
            "peak": peak,
            "zero_crossing_rate": zero_crossing_rate,
            "spectral_centroid": centroid_hz / sample_rate,
            "spectral_bandwidth": bandwidth_hz / sample_rate,
            "spectral_flatness": flatness,
            "spectral_flux": spectral_flux,
            "low_energy_ratio": energy_ratio(0.0, 300.0),
            "mid_energy_ratio": energy_ratio(300.0, 3_000.0),
            "high_energy_ratio": energy_ratio(3_000.0, None),
            "f0_normalized": f0_hz / 350.0,
            "voicing_confidence": voicing,
            "log_rms_delta": delta,
        }
        self.previous_log_rms = log_rms
        self.previous_spectrum = normalized.copy()
        return features

    @staticmethod
    def _estimate_f0(samples: np.ndarray, sample_rate: int) -> tuple[float, float]:
        centered = samples - np.mean(samples)
        energy = float(np.dot(centered, centered))
        minimum_lag = max(1, math.ceil(sample_rate / 350.0))
        maximum_lag = min(samples.size - 1, math.floor(sample_rate / 70.0))
        if energy <= 1e-24 or maximum_lag < minimum_lag:
            return 0.0, 0.0
        correlations = np.correlate(centered, centered, mode="full")[samples.size - 1 :]
        search = correlations[minimum_lag : maximum_lag + 1]
        lag = minimum_lag + int(np.argmax(search))
        confidence = float(np.clip(correlations[lag] / correlations[0], 0.0, 1.0))
        if 1 <= lag < correlations.size - 1:
            left, middle, right = correlations[lag - 1 : lag + 2]
            denominator = left - 2.0 * middle + right
            if abs(denominator) > 1e-24:
                lag += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
        return float(sample_rate / lag), confidence


__all__ = ["AudioFeatureExtractor", "FEATURE_NAMES"]
