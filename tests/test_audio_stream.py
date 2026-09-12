from __future__ import annotations

import json

import numpy as np
import pytest

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_probe import DeterministicAudioCoupler
from demian_v1.audio_stream import AudioStreamConfig, IncrementalAudioProcessor


def test_stream_rejects_hop_larger_than_frame() -> None:
    with pytest.raises(ValueError, match="audio_stream_frame_config_invalid"):
        AudioStreamConfig(sample_rate=8_000, frame_size=200, hop_size=201)


def _samples() -> np.ndarray:
    return np.asarray([np.sin(index / 5.0) * 0.7 for index in range(37)], dtype=np.float32)


def _processor() -> IncrementalAudioProcessor:
    return IncrementalAudioProcessor(
        AudioStreamConfig(sample_rate=1_000, frame_size=10, hop_size=4, strength=0.15, segment_id="test"),
        runtime=DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4)),
        coupler=DeterministicAudioCoupler(hidden_size=8, projection_seed=6),
    )


def test_feed_defers_partial_frames_without_boundary_padding() -> None:
    processor = _processor()

    assert processor.feed(_samples()[:9]) == []
    rows = processor.feed(_samples()[9:10])

    assert [row["sample_offset"] for row in rows] == [0]
    assert processor.pending_samples.tolist() == _samples()[4:10].tolist()


def test_random_chunking_matches_offline_frames_after_one_final_pad() -> None:
    samples = _samples()
    streamed = _processor()
    rows: list[dict[str, object]] = []
    for chunk in (samples[:3], samples[3:12], samples[12:13], samples[13:26], samples[26:]):
        rows.extend(streamed.feed(chunk))
    rows.extend(streamed.finalize())

    offline = _processor()
    expected = offline.feed(samples) + offline.finalize()

    assert json.dumps(rows, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    assert [row["sample_offset"] for row in rows] == [0, 4, 8, 12, 16, 20, 24, 28]
    assert streamed.pending_samples.size == 0
    assert streamed.finalize() == []


def test_json_snapshot_resume_matches_uninterrupted_processing() -> None:
    samples = _samples()
    uninterrupted = _processor()
    expected = uninterrupted.feed(samples) + uninterrupted.finalize()

    first = _processor()
    actual = first.feed(samples[:17])
    snapshot = json.loads(json.dumps(first.snapshot()))
    resumed = _processor()
    resumed.restore(snapshot)
    actual.extend(resumed.feed(samples[17:]))
    actual.extend(resumed.finalize())

    assert actual == expected


def test_restore_rejects_incompatible_config_before_mutating_state() -> None:
    processor = _processor()
    processor.feed(_samples()[:13])
    before = processor.snapshot()
    incompatible = dict(before)
    incompatible["config"] = {**before["config"], "hop_size": 5}

    with pytest.raises(ValueError, match="audio_stream_config_mismatch"):
        processor.restore(incompatible)

    assert processor.snapshot() == before


def test_runtime_restore_rejects_different_construction_config() -> None:
    source = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4))
    target = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=5))

    with pytest.raises(ValueError, match="demian_v1_config_mismatch"):
        target.restore(source.snapshot())


def test_coupler_snapshot_carries_and_validates_projection_identity() -> None:
    source = DeterministicAudioCoupler(hidden_size=8, projection_seed=6, output_gain=0.5)
    target = DeterministicAudioCoupler(hidden_size=8, projection_seed=7, output_gain=0.5)

    snapshot = source.snapshot()
    assert snapshot["projection_seed"] == 6
    assert snapshot["output_gain"] == 0.5
    assert len(snapshot["weights"]) == 8
    with pytest.raises(ValueError, match="audio_coupler_config_mismatch"):
        target.restore(snapshot)
