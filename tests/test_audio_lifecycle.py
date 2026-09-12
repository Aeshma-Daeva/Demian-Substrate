from __future__ import annotations

import json

import numpy as np
import pytest

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_lifecycle import SegmentedLiveAudioSession, SegmentedState
from demian_v1.audio_stream import AudioStreamConfig, DeterministicAudioCoupler, IncrementalAudioProcessor


class Capture:
    def start(self, block, fault) -> None:  # type: ignore[no-untyped-def]
        self.block, self.fault = block, fault
    def stop(self) -> None:
        pass


def test_pause_resume_closes_segment_preserves_runtime_and_resets_acoustic_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4))
    captures: list[Capture] = []

    def processor(segment_id: str, frame_index: int) -> IncrementalAudioProcessor:
        return IncrementalAudioProcessor(
            AudioStreamConfig(1_000, 4, 4, segment_id=segment_id, frame_index_start=frame_index),
            runtime=runtime, coupler=DeterministicAudioCoupler(8, projection_seed=6),
        )
    def capture() -> Capture:
        value = Capture()
        captures.append(value)
        return value

    session = SegmentedLiveAudioSession(processor, capture, output_dir=tmp_path, capacity_samples=20, session_id="run")
    session.start()
    captures[0].block(np.ones(6, dtype=np.float32))
    session.drain()
    session.pause()
    checkpoint = session.boundary_checkpoint()
    session.resume()
    captures[1].block(np.zeros(4, dtype=np.float32))
    session.drain()
    session.stop()

    first = [json.loads(line) for line in (tmp_path / "segments" / "run-000" / "frames.jsonl").read_text().splitlines()]
    second = [json.loads(line) for line in (tmp_path / "segments" / "run-001" / "frames.jsonl").read_text().splitlines()]
    assert session.state is SegmentedState.STOPPED
    assert [row["frame_index"] for row in first + second] == list(range(len(first) + len(second)))
    assert second[0]["sample_offset"] == 0
    assert first[-1]["padded_sample_count"] == 2
    assert checkpoint["schema_version"] == 1
    assert "pending_samples" not in json.dumps(checkpoint)
    assert "extractor" not in json.dumps(checkpoint)
    assert checkpoint["lineage"][-1]["segment_id"] == "run-000"


def test_checkpoint_requires_closed_segment_and_restore_is_new_segment(tmp_path) -> None:  # type: ignore[no-untyped-def]
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4))
    captures: list[Capture] = []
    def processor(segment_id: str, frame_index: int) -> IncrementalAudioProcessor:
        return IncrementalAudioProcessor(AudioStreamConfig(1_000, 4, 4, segment_id=segment_id, frame_index_start=frame_index), runtime=runtime, coupler=DeterministicAudioCoupler(8, projection_seed=6))
    def capture() -> Capture:
        value = Capture()
        captures.append(value)
        return value
    session = SegmentedLiveAudioSession(processor, capture, output_dir=tmp_path, capacity_samples=20, session_id="restore")
    session.start()
    with pytest.raises(ValueError, match="segment_checkpoint_not_closed"):
        session.boundary_checkpoint()
    session.pause()
    checkpoint = session.boundary_checkpoint()
    restored_runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4))
    restored = SegmentedLiveAudioSession.restore(checkpoint, processor_factory=lambda sid, index: IncrementalAudioProcessor(AudioStreamConfig(1_000, 4, 4, segment_id=sid, frame_index_start=index), runtime=restored_runtime, coupler=DeterministicAudioCoupler(8, projection_seed=6)), capture_factory=capture, output_dir=tmp_path / "restored", capacity_samples=20)

    assert restored.state is SegmentedState.PAUSED
    restored.resume()
    assert restored.current_segment_id == "restore-001"
