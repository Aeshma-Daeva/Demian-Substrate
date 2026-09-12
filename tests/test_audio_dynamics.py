from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from demian_v1.audio_dynamics import AudioFeatureExtractor


def _tone(frequency: float, *, amplitude: float = 0.8, sample_rate: int = 8_000) -> np.ndarray:
    time = np.arange(800, dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2.0 * math.pi * frequency * time)).astype(np.float32)


def test_features_separate_pitch_brightness_and_amplitude() -> None:
    low = AudioFeatureExtractor().extract(_tone(120.0), 8_000)
    high = AudioFeatureExtractor().extract(_tone(1_200.0), 8_000)
    quiet = AudioFeatureExtractor().extract(_tone(120.0, amplitude=0.2), 8_000)

    assert low["f0_normalized"] * 350.0 == pytest.approx(120.0, abs=2.0)
    assert low["spectral_centroid"] == pytest.approx(0.015, abs=0.005)
    assert high["spectral_centroid"] == pytest.approx(0.15, abs=0.01)
    assert low["peak"] == pytest.approx(0.8, abs=0.01)
    assert quiet["peak"] == pytest.approx(0.2, abs=0.01)


def test_temporal_features_survive_snapshot_restore() -> None:
    first = _tone(120.0, amplitude=0.2)
    second = _tone(300.0, amplitude=0.8)
    continuous = AudioFeatureExtractor()
    continuous.extract(first, 8_000)
    snapshot = continuous.snapshot()
    expected = continuous.extract(second, 8_000)

    resumed = AudioFeatureExtractor()
    resumed.restore(snapshot)
    actual = resumed.extract(second, 8_000)

    assert actual["spectral_flux"] == pytest.approx(expected["spectral_flux"])
    assert actual["log_rms_delta"] == pytest.approx(expected["log_rms_delta"])


def test_silence_and_two_sample_frames_remain_finite() -> None:
    extractor = AudioFeatureExtractor()

    for frame in (np.zeros(200, dtype=np.float32), np.array([-0.5, 0.5], dtype=np.float32)):
        features = extractor.extract(frame, 8_000)
        assert all(math.isfinite(value) for value in features.values())


def test_invalid_snapshot_values_are_rejected() -> None:
    extractor = AudioFeatureExtractor()

    with pytest.raises(ValueError, match="audio_feature_snapshot_invalid"):
        extractor.restore({"previous_log_rms": float("nan"), "previous_spectrum": []})
    with pytest.raises(ValueError, match="audio_feature_snapshot_invalid"):
        extractor.restore({"previous_log_rms": True, "previous_spectrum": []})


def test_seeded_projection_is_deterministic_bounded_and_signal_sensitive() -> None:
    from demian_v1.audio_probe import DeterministicAudioCoupler

    low = DeterministicAudioCoupler(hidden_size=8, projection_seed=17)
    same = DeterministicAudioCoupler(hidden_size=8, projection_seed=17)
    other_signal = DeterministicAudioCoupler(hidden_size=8, projection_seed=17)

    _, first = low.encode(_tone(120.0), 8_000)
    _, repeated = same.encode(_tone(120.0), 8_000)
    _, different = other_signal.encode(_tone(300.0), 8_000)

    assert torch.equal(first, repeated)
    assert torch.all(first.abs() <= 1.0)
    assert not torch.equal(first, different)
