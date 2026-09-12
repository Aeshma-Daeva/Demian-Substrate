"""Segmented pause/resume orchestration over the bounded live-audio session."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import json
from pathlib import Path

from demian_v1.audio_stream import IncrementalAudioProcessor
from demian_v1.live_audio import CaptureBackend, LiveAudioSession, LiveAudioState
from demian_v1.runtime import DemianV1Runtime


class SegmentedState(str, Enum):
    NEW = "NEW"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"


ProcessorFactory = Callable[[str, int], IncrementalAudioProcessor]
CaptureFactory = Callable[[], CaptureBackend]


class SegmentedLiveAudioSession:
    """Link independently finalized segments while retaining recurrent runtime state.

    A pause never carries acoustic framing or extractor history into the next
    segment. The processor factory is responsible for constructing a fresh
    coupler/extractor around the preserved runtime.
    """

    def __init__(self, processor_factory: ProcessorFactory, capture_factory: CaptureFactory, *, output_dir: Path,
                 capacity_samples: int, session_id: str = "session") -> None:
        if not session_id:
            raise ValueError("segmented_audio_session_id_invalid")
        self.processor_factory, self.capture_factory = processor_factory, capture_factory
        self.output_dir, self.capacity_samples, self.session_id = Path(output_dir), capacity_samples, session_id
        self.state = SegmentedState.NEW
        self._segment_number = 0
        self._frame_index = 0
        self._current: LiveAudioSession | None = None
        self._lineage: list[dict[str, object]] = []

    @property
    def current_segment_id(self) -> str | None:
        return None if self._current is None else self._current.processor.config.segment_id

    def _event(self, event: str, **values: object) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "lifecycle.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps({"event": event, **values}, allow_nan=False) + "\n")

    def _open_segment(self) -> None:
        segment_id = f"{self.session_id}-{self._segment_number:03d}"
        processor = self.processor_factory(segment_id, self._frame_index)
        capture = self.capture_factory()
        current = LiveAudioSession(processor, capture, output_dir=self.output_dir / "segments" / segment_id,
                                   capacity_samples=self.capacity_samples)
        current.start()
        self._current = current
        if current.state is LiveAudioState.FAILED:
            self.state = SegmentedState.FAILED
        else:
            self.state = SegmentedState.RUNNING
            self._event("segment_started", segment_id=segment_id, frame_index_start=self._frame_index,
                        parent_segment_id=self._lineage[-1]["segment_id"] if self._lineage else None)

    def start(self) -> None:
        if self.state is not SegmentedState.NEW:
            raise ValueError("segmented_audio_invalid_state")
        self._open_segment()

    def drain(self) -> None:
        if self.state is not SegmentedState.RUNNING or self._current is None:
            return
        self._current.drain()
        if self._current.state is LiveAudioState.FAILED:
            self.state = SegmentedState.FAILED

    def _close_current(self, terminal_event: str) -> None:
        if self._current is None:
            raise ValueError("segmented_audio_no_active_segment")
        current = self._current
        current.stop()
        if current.state is LiveAudioState.FAILED:
            self.state = SegmentedState.FAILED
            return
        snapshot = current.processor.snapshot()
        self._frame_index = int(snapshot["frame_index"])
        record = {
            "segment_id": current.processor.config.segment_id,
            "accepted_samples": current.summary["accepted_samples"],
            "frame_index_end": self._frame_index,
            "state": current.state.value,
        }
        self._lineage.append(record)
        self._event(terminal_event, **record)
        self._current = None
        self._segment_number += 1

    def pause(self) -> None:
        if self.state is SegmentedState.PAUSED:
            return
        if self.state is not SegmentedState.RUNNING:
            raise ValueError("segmented_audio_invalid_state")
        self._close_current("segment_paused")
        if self.state is not SegmentedState.FAILED:
            self.state = SegmentedState.PAUSED

    def resume(self) -> None:
        if self.state is not SegmentedState.PAUSED:
            raise ValueError("segmented_audio_invalid_state")
        self._open_segment()

    def stop(self) -> None:
        if self.state in (SegmentedState.STOPPED, SegmentedState.FAILED):
            return
        if self.state is SegmentedState.PAUSED:
            self.state = SegmentedState.STOPPED
            self._event("stopped", lineage=self._lineage)
            return
        if self.state is not SegmentedState.RUNNING:
            raise ValueError("segmented_audio_invalid_state")
        self._close_current("segment_stopped")
        if self.state is not SegmentedState.FAILED:
            self.state = SegmentedState.STOPPED
            self._event("stopped", lineage=self._lineage)

    def boundary_checkpoint(self) -> dict[str, object]:
        """Return only closed-boundary state, never PCM/pending framing/history."""
        if self.state not in (SegmentedState.PAUSED, SegmentedState.STOPPED):
            raise ValueError("segment_checkpoint_not_closed")
        if not self._lineage:
            raise ValueError("segment_checkpoint_empty")
        processor = self.processor_factory(f"{self.session_id}-checkpoint", self._frame_index)
        coupler = processor.coupler
        return {
            "checkpoint_id": "demian-v1-segmented-live-audio",
            "schema_version": 1,
            "valid": self.state is not SegmentedState.FAILED,
            "session_id": self.session_id,
            "state": self.state.value,
            "frame_index": self._frame_index,
            "lineage": list(self._lineage),
            "runtime": processor.runtime.snapshot().to_dict(),
            "projection": {"projection_seed": coupler.projection_seed, "output_gain": coupler.output_gain,
                           "weights": coupler.weights.detach().cpu().tolist()},
            "processor_config": {"sample_rate": processor.config.sample_rate, "frame_size": processor.config.frame_size,
                                 "hop_size": processor.config.hop_size, "strength": processor.config.strength},
        }

    @classmethod
    def restore(cls, checkpoint: dict[str, object], *, processor_factory: ProcessorFactory,
                capture_factory: CaptureFactory, output_dir: Path, capacity_samples: int) -> "SegmentedLiveAudioSession":
        """Validate completely before restoring runtime; resumed work opens a new segment."""
        try:
            if checkpoint["checkpoint_id"] != "demian-v1-segmented-live-audio" or checkpoint["schema_version"] != 1:
                raise ValueError
            if checkpoint["valid"] is not True:
                raise ValueError
            if checkpoint["state"] not in (SegmentedState.PAUSED.value, SegmentedState.STOPPED.value):
                raise ValueError
            session_id, frame_index, lineage, runtime_data, projection = (
                checkpoint["session_id"], checkpoint["frame_index"], checkpoint["lineage"], checkpoint["runtime"], checkpoint["projection"])
            if not isinstance(session_id, str) or type(frame_index) is not int or frame_index < 0 or not lineage or not isinstance(lineage, list):
                raise ValueError
            if not isinstance(runtime_data, dict) or not isinstance(projection, dict):
                raise ValueError
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("segmented_audio_checkpoint_invalid") from error
        probe = processor_factory(f"{session_id}-checkpoint", frame_index)
        identity = {"projection_seed": probe.coupler.projection_seed, "output_gain": probe.coupler.output_gain,
                    "weights": probe.coupler.weights.detach().cpu().tolist()}
        if projection != identity:
            raise ValueError("segmented_audio_checkpoint_projection_mismatch")
        trial = DemianV1Runtime(probe.runtime.config)
        try:
            trial.restore(runtime_data)
        except ValueError as error:
            raise ValueError("segmented_audio_checkpoint_invalid") from error
        probe.runtime.restore(runtime_data)
        restored = cls(processor_factory, capture_factory, output_dir=output_dir, capacity_samples=capacity_samples,
                       session_id=session_id)
        restored.state, restored._frame_index, restored._lineage = SegmentedState.PAUSED, frame_index, lineage
        restored._segment_number = len(lineage)
        return restored


__all__ = ["SegmentedLiveAudioSession", "SegmentedState"]
