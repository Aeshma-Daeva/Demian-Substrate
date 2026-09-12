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


def test_feed_rejects_stereo_input_instead_of_flattening_it() -> None:
    processor = _processor()

    with pytest.raises(ValueError, match="audio_stream_samples_must_be_mono_1d"):
        processor.feed(np.zeros((4, 2), dtype=np.float32))

    assert processor.pending_samples.size == 0


def test_trace_identifies_real_and_final_padding_samples() -> None:
    processor = _processor()

    assert processor.feed(_samples()[:9]) == []
    row = processor.finalize()[0]

    assert row["real_sample_count"] == 9
    assert row["padded_sample_count"] == 1


@pytest.mark.parametrize("partition_seed", [0, 4, 91])
def test_seeded_random_chunk_partitions_match_offline(partition_seed: int) -> None:
    samples = _samples()
    random = np.random.default_rng(partition_seed)
    streamed = _processor()
    rows: list[dict[str, object]] = []
    index = 0
    while index < samples.size:
        next_index = min(samples.size, index + int(random.integers(1, 8)))
        rows.extend(streamed.feed(samples[index:next_index]))
        index = next_index
    rows.extend(streamed.finalize())

    offline = _processor()
    assert rows == offline.feed(samples) + offline.finalize()


@pytest.mark.parametrize(("samples", "expected"), [
    (np.empty(0, dtype=np.float32), []),
    (np.ones(10, dtype=np.float32), [(10, 0), (6, 4)]),
    (np.ones(14, dtype=np.float32), [(10, 0), (10, 0), (6, 4)]),
    (np.ones(9, dtype=np.float32), [(9, 1)]),
])
def test_finalization_real_and_padded_counts_at_boundaries(
    samples: np.ndarray, expected: list[tuple[int, int]],
) -> None:
    processor = _processor()
    rows = processor.feed(samples) + processor.finalize()

    assert [(row["real_sample_count"], row["padded_sample_count"]) for row in rows] == expected


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


@pytest.mark.parametrize(("field", "value"), [
    ("finalized", 1), ("pending_offset", -1), ("frame_index", -1),
])
def test_restore_rejects_invalid_scalar_state_before_mutating_state(field: str, value: object) -> None:
    processor = _processor()
    processor.feed(_samples()[:13])
    before = processor.snapshot()
    malformed = dict(before)
    malformed[field] = value

    with pytest.raises(ValueError, match="audio_stream_snapshot_invalid"):
        processor.restore(malformed)

    assert processor.snapshot() == before


def test_restore_rejects_finalized_snapshot_with_pending_pcm_before_mutating_state() -> None:
    processor = _processor()
    processor.feed(_samples()[:13])
    before = processor.snapshot()
    malformed = dict(before)
    malformed["finalized"] = True

    with pytest.raises(ValueError, match="audio_stream_snapshot_invalid"):
        processor.restore(malformed)

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
