from __future__ import annotations

import json

import numpy as np
import pytest

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_stream import AudioStreamConfig, DeterministicAudioCoupler, IncrementalAudioProcessor
from demian_v1.live_audio import LiveAudioSession, LiveAudioState, run_timed_session


class FakeCapture:
    def __init__(self) -> None:
        self.callback = None
        self.fault_callback = None
        self.started = False
        self.stopped = False

    def start(self, on_block, on_fault) -> None:  # type: ignore[no-untyped-def]
        self.callback = on_block
        self.fault_callback = on_fault
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def emit(self, samples: np.ndarray) -> None:
        assert self.callback is not None
        self.callback(samples)

    def fail(self, reason: str) -> None:
        assert self.fault_callback is not None
        self.fault_callback(reason)


def _processor() -> IncrementalAudioProcessor:
    return IncrementalAudioProcessor(
        AudioStreamConfig(sample_rate=1_000, frame_size=10, hop_size=4, segment_id="live"),
        runtime=DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4)),
        coupler=DeterministicAudioCoupler(hidden_size=8, projection_seed=6),
    )


def test_session_accepts_ordered_prefix_and_writes_separate_jsonl(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=20)

    session.start()
    capture.emit(np.arange(10, dtype=np.float32))
    capture.emit(np.arange(10, 20, dtype=np.float32))
    session.drain()
    session.stop()

    assert session.state is LiveAudioState.STOPPED
    assert session.summary["accepted_samples"] == 20
    assert session.summary["experiment_valid"] is True
    frames = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [frame["sample_offset"] for frame in frames] == [0, 4, 8, 12]
    assert events[0]["event"] == "started"
    assert events[-1]["event"] == "stopped"


def test_session_overflow_latches_terminal_fault_without_dropping_old_blocks(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=10)
    session.start()
    capture.emit(np.ones(10, dtype=np.float32))
    capture.emit(np.ones(1, dtype=np.float32))

    assert session.state is LiveAudioState.FAILED
    assert session.summary["failure"]["kind"] == "buffer_overflow"
    assert session.summary["accepted_samples"] == 10
    assert capture.stopped is True


def test_session_rejects_oversize_callback_before_accepting_any_samples(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=10)
    session.start()
    capture.emit(np.ones(11, dtype=np.float32))

    assert session.state is LiveAudioState.FAILED
    assert session.summary["failure"]["kind"] == "oversize_callback_block"
    assert session.summary["accepted_samples"] == 0


def test_session_capture_fault_is_terminal_even_when_queue_is_not_full(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=20)
    session.start()
    capture.emit(np.ones(2, dtype=np.float32))
    capture.fail("device_lost")

    assert session.state is LiveAudioState.FAILED
    assert session.summary["failure"] == {"kind": "capture_fault", "detail": "device_lost"}


def test_live_checkpoint_excludes_pending_raw_pcm(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=20)
    session.start()
    capture.emit(np.arange(3, dtype=np.float32))

    checkpoint = session.boundary_checkpoint()

    assert "pending_samples" not in json.dumps(checkpoint)
    assert checkpoint["state"] == "RUNNING"
    assert checkpoint["accepted_samples"] == 3


def test_session_cannot_start_again_after_terminal_stop(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=20)
    session.start()
    session.stop()

    with pytest.raises(ValueError, match="live_audio_invalid_state"):
        session.start()


def test_timed_session_stops_cleanly_with_fake_clock(tmp_path) -> None:  # type: ignore[no-untyped-def]
    capture = FakeCapture()
    session = LiveAudioSession(_processor(), capture, output_dir=tmp_path, capacity_samples=20)
    moments = iter([0.0, 0.0, 0.4, 1.0])

    run_timed_session(session, duration_seconds=1.0, now=lambda: next(moments), sleep=lambda _: None)

    assert session.state is LiveAudioState.STOPPED
    assert capture.started is True
    assert capture.stopped is True
