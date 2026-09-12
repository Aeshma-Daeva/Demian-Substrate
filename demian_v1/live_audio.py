"""Local live-audio session boundary; capture never performs analysis or writes."""

from __future__ import annotations

from collections import deque
from enum import Enum
import json
from pathlib import Path
import time
from typing import Callable, Protocol

import numpy as np

from demian_v1.audio_stream import IncrementalAudioProcessor


class CaptureBackend(Protocol):
    def start(self, on_block: Callable[[np.ndarray], None], on_fault: Callable[[str], None]) -> None: ...
    def stop(self) -> None: ...


class LiveAudioState(str, Enum):
    NEW = "NEW"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


class LiveAudioSession:
    """Bounded capture handoff with terminal fault latching and JSONL evidence."""

    def __init__(
        self, processor: IncrementalAudioProcessor, capture: CaptureBackend, *, output_dir: Path,
        capacity_samples: int,
    ) -> None:
        if capacity_samples <= 0:
            raise ValueError("live_audio_capacity_invalid")
        self.processor, self.capture = processor, capture
        self.output_dir, self.capacity_samples = Path(output_dir), capacity_samples
        self.state = LiveAudioState.NEW
        self._queue: deque[np.ndarray] = deque()
        self._queued_samples = 0
        self._accepted_samples = 0
        self._peak_occupancy = 0
        self._failure: dict[str, str] | None = None
        self._pending_failure: dict[str, str] | None = None
        self._accepting = False
        self._frames_file = None
        self._events_file = None

    @property
    def summary(self) -> dict[str, object]:
        return {
            "state": self.state.value, "experiment_valid": self._failure is None and self.state is LiveAudioState.STOPPED,
            "accepted_samples": self._accepted_samples, "capacity_samples": self.capacity_samples,
            "peak_occupancy_samples": self._peak_occupancy, "failure": self._failure,
            "parameters": {"sample_rate": self.processor.config.sample_rate, "frame_size": self.processor.config.frame_size,
                           "hop_size": self.processor.config.hop_size, "segment_id": self.processor.config.segment_id},
        }

    def _event(self, event: str, **values: object) -> None:
        assert self._events_file is not None
        self._events_file.write(json.dumps({"event": event, **values}, allow_nan=False) + "\n")
        self._events_file.flush()

    def start(self) -> None:
        if self.state is not LiveAudioState.NEW:
            raise ValueError("live_audio_invalid_state")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._frames_file = (self.output_dir / "frames.jsonl").open("x", encoding="utf-8")
            self._events_file = (self.output_dir / "events.jsonl").open("x", encoding="utf-8")
            self.state = LiveAudioState.STARTING
            self._event("started", **self.summary)
            self._accepting = True
            self.capture.start(self.accept_block, self.capture_fault)
            if self._pending_failure is None:
                self.state = LiveAudioState.RUNNING
            else:
                self._complete_pending_failure()
        except Exception as error:
            self._fail("start_failure", str(error))

    def _latch_failure(self, kind: str, detail: str) -> None:
        """Record a terminal callback fault without capture control or file I/O."""
        if self._pending_failure is None and self._failure is None:
            self._pending_failure = {"kind": kind, "detail": detail}

    def _fail(self, kind: str, detail: str) -> None:
        if self.state is LiveAudioState.FAILED:
            return
        self._failure = {"kind": kind, "detail": detail}
        self.state = LiveAudioState.FAILED
        self._accepting = False
        try:
            self.capture.stop()
        except Exception:
            pass
        if self._events_file is not None:
            self._event("failed", failure=self._failure)
        self._close_files()

    def _close_files(self) -> None:
        for handle_name in ("_frames_file", "_events_file"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)

    def accept_block(self, samples: np.ndarray) -> None:
        """Callback entry: copy and enqueue only; never processes or writes frames."""
        if not self._accepting or self.state not in (
            LiveAudioState.STARTING,
            LiveAudioState.RUNNING,
            LiveAudioState.DRAINING,
        ) or self._pending_failure is not None:
            return
        raw = np.asarray(samples)
        if raw.ndim != 1:
            self._latch_failure("invalid_callback_block", "samples_must_be_mono_1d")
            return
        try:
            block = raw.astype(np.float32, copy=True)
        except (TypeError, ValueError):
            self._latch_failure("invalid_callback_block", "samples_not_float_compatible")
            return
        if not np.all(np.isfinite(block)):
            self._latch_failure("invalid_callback_block", "samples_not_finite")
        elif block.size > self.capacity_samples:
            self._latch_failure("oversize_callback_block", "block_exceeds_capacity")
        elif self._queued_samples + block.size > self.capacity_samples:
            self._latch_failure("buffer_overflow", "bounded_handoff_full")
        else:
            self._queue.append(block)
            self._queued_samples += block.size
            self._accepted_samples += block.size
            self._peak_occupancy = max(self._peak_occupancy, self._queued_samples)

    def capture_fault(self, detail: str) -> None:
        self._latch_failure("capture_fault", str(detail))

    def _write_queued_frames(self) -> None:
        while self._queue:
            block = self._queue.popleft()
            self._queued_samples -= block.size
            for row in self.processor.feed(block):
                assert self._frames_file is not None
                self._frames_file.write(json.dumps(row, allow_nan=False) + "\n")
        assert self._frames_file is not None
        self._frames_file.flush()

    def _complete_pending_failure(self) -> None:
        pending = self._pending_failure
        if pending is None:
            return
        self._accepting = False
        self.state = LiveAudioState.DRAINING
        try:
            self.capture.stop()
            self._write_queued_frames()
        except Exception as error:
            pending = {"kind": "processing_or_writer_fault", "detail": str(error)}
        self._pending_failure = None
        self._failure = pending
        self.state = LiveAudioState.FAILED
        if self._events_file is not None:
            self._event("failed", failure=self._failure)
        self._close_files()

    def drain(self) -> None:
        if self.state is not LiveAudioState.RUNNING:
            return
        if self._pending_failure is not None:
            self._complete_pending_failure()
            return
        self.state = LiveAudioState.DRAINING
        try:
            self._write_queued_frames()
            self.state = LiveAudioState.RUNNING
        except Exception as error:
            self._fail("processing_or_writer_fault", str(error))

    def stop(self) -> None:
        if self.state in (LiveAudioState.FAILED, LiveAudioState.STOPPED):
            return
        if self.state is not LiveAudioState.RUNNING:
            raise ValueError("live_audio_invalid_state")
        try:
            self._accepting = False
            self.capture.stop()
            self.drain()
            if self.state is LiveAudioState.FAILED:
                return
            assert self._frames_file is not None
            for row in self.processor.finalize():
                self._frames_file.write(json.dumps(row, allow_nan=False) + "\n")
            self._frames_file.flush()
            self.state = LiveAudioState.STOPPED
            self._event("stopped", **self.summary)
            self._close_files()
        except Exception as error:
            self._fail("processing_or_writer_fault", str(error))

    def boundary_checkpoint(self) -> dict[str, object]:
        """Derived live boundary state only: intentionally excludes queued/pending PCM."""
        return {"live_audio_id": "demian-v1-live-audio", "state": self.state.value,
                "accepted_samples": self._accepted_samples, "capacity_samples": self.capacity_samples,
                "peak_occupancy_samples": self._peak_occupancy, "failure": self._failure,
                "processor_config": self.processor.snapshot()["config"]}


def run_timed_session(
    session: LiveAudioSession, *, duration_seconds: float, now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Run a timed session while keeping callback work confined to capture handoff."""
    if duration_seconds <= 0:
        raise ValueError("live_audio_duration_invalid")
    session.start()
    deadline = now() + duration_seconds
    while session.state is LiveAudioState.RUNNING and now() < deadline:
        session.drain()
        sleep(0.01)
    if session.state is LiveAudioState.RUNNING:
        session.stop()


__all__ = ["CaptureBackend", "LiveAudioSession", "LiveAudioState", "run_timed_session"]
