"""Deterministic incremental audio framing for local acoustic probes.

This module deliberately has no capture, transport, or semantic interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from demian_v1.audio_dynamics import FEATURE_NAMES, AudioFeatureExtractor
from demian_v1.runtime import DemianV1Runtime
from development.demian_v1_gate_state import V1_CHANNELS


class DeterministicAudioCoupler:
    """Seeded, bounded acoustic-feature projection into the runtime boundary."""

    def __init__(self, hidden_size: int, *, projection_seed: int = 0, output_gain: float = 0.75,
                 extractor: AudioFeatureExtractor | None = None) -> None:
        if hidden_size < 2:
            raise ValueError("audio_coupler_hidden_size_must_be_at_least_two")
        if not math.isfinite(output_gain):
            raise ValueError("audio_coupler_output_gain_invalid")
        self.projection_seed = int(projection_seed)
        self.output_gain = float(output_gain)
        generator = np.random.default_rng(self.projection_seed)
        weights = generator.normal(0.0, 1.0 / math.sqrt(len(FEATURE_NAMES)), (hidden_size, len(FEATURE_NAMES)))
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.extractor = extractor or AudioFeatureExtractor()

    def encode(self, frame: np.ndarray, sample_rate: int) -> tuple[dict[str, float], torch.Tensor]:
        features = self.extractor.extract(frame, sample_rate)
        values = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
        values[0] = np.clip((values[0] + 6.0) / 6.0, -1.0, 1.0)
        values[6] = np.clip(values[6] * 10.0, 0.0, 1.0)
        values[12] = np.clip(values[12], -1.0, 1.0)
        return features, torch.tanh(self.output_gain * (self.weights @ torch.from_numpy(values)))

    def snapshot(self) -> dict[str, object]:
        return {"projection_seed": self.projection_seed, "output_gain": self.output_gain,
                "weights": self.weights.detach().cpu().tolist(), "extractor": self.extractor.snapshot()}

    def restore(self, snapshot: dict[str, object]) -> None:
        try:
            seed = int(snapshot["projection_seed"])
            gain = float(snapshot["output_gain"])
            weights = torch.tensor(snapshot["weights"], dtype=torch.float32)
            extractor = snapshot["extractor"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("audio_coupler_snapshot_invalid") from error
        if (seed != self.projection_seed or gain != self.output_gain or weights.shape != self.weights.shape
                or not torch.equal(weights, self.weights)):
            raise ValueError("audio_coupler_config_mismatch")
        if not isinstance(extractor, dict):
            raise ValueError("audio_coupler_snapshot_invalid")
        self.extractor.restore(extractor)


@dataclass(frozen=True)
class AudioStreamConfig:
    sample_rate: int
    frame_size: int
    hop_size: int
    strength: float = 0.15
    segment_id: str = "default"
    frame_index_start: int = 0
    sample_offset_start: int = 0

    def __post_init__(self) -> None:
        if (
            self.sample_rate <= 0
            or self.frame_size <= 0
            or self.hop_size <= 0
            or self.hop_size > self.frame_size
        ):
            raise ValueError("audio_stream_frame_config_invalid")
        if not math.isfinite(self.strength) or not self.segment_id:
            raise ValueError("audio_stream_config_invalid")


class IncrementalAudioProcessor:
    """Frame arbitrary mono float PCM chunks without chunk-boundary silence."""

    def __init__(self, config: AudioStreamConfig, *, runtime: DemianV1Runtime,
                 coupler: DeterministicAudioCoupler) -> None:
        if runtime.config.hidden_size != coupler.weights.shape[0]:
            raise ValueError("audio_stream_hidden_size_mismatch")
        self.config = config
        self.runtime = runtime
        self.coupler = coupler
        self.pending_samples = np.empty(0, dtype=np.float32)
        self._pending_offset = config.sample_offset_start
        self._frame_index = config.frame_index_start
        self._finalized = False

    def process_frame(self, frame: np.ndarray, sample_offset: int, *, real_sample_count: int) -> dict[str, object]:
        features, coupling = self.coupler.encode(frame, self.config.sample_rate)
        step = self.runtime.step(coupling, strength=self.config.strength)
        row = {
            "segment_id": self.config.segment_id, "frame_index": self._frame_index,
            "sample_offset": sample_offset, "time_seconds": sample_offset / self.config.sample_rate,
            "real_sample_count": real_sample_count,
            "padded_sample_count": self.config.frame_size - real_sample_count,
            "features": features, "coupling": coupling.detach().cpu().tolist(), "surface": step["surface"],
            "channel_norms": {channel: float(torch.linalg.vector_norm(state).item())
                              for channel, state in zip(V1_CHANNELS, self.runtime.state, strict=True)},
            "metrics": step["metrics"],
        }
        self._frame_index += 1
        return row

    def feed(self, samples: np.ndarray) -> list[dict[str, object]]:
        if self._finalized:
            raise ValueError("audio_stream_already_finalized")
        raw = np.asarray(samples)
        if raw.ndim != 1:
            raise ValueError("audio_stream_samples_must_be_mono_1d")
        try:
            chunk = raw.astype(np.float32, copy=False)
        except (TypeError, ValueError) as error:
            raise ValueError("audio_stream_samples_invalid") from error
        if not np.all(np.isfinite(chunk)):
            raise ValueError("audio_stream_samples_invalid")
        self.pending_samples = np.concatenate((self.pending_samples, chunk))
        rows: list[dict[str, object]] = []
        while self.pending_samples.size >= self.config.frame_size:
            rows.append(self.process_frame(
                self.pending_samples[:self.config.frame_size], self._pending_offset,
                real_sample_count=self.config.frame_size,
            ))
            self.pending_samples = self.pending_samples[self.config.hop_size:].copy()
            self._pending_offset += self.config.hop_size
        return rows

    def finalize(self) -> list[dict[str, object]]:
        if self._finalized:
            return []
        self._finalized = True
        if not self.pending_samples.size:
            return []
        frame = np.pad(self.pending_samples, (0, self.config.frame_size - self.pending_samples.size))
        row = self.process_frame(frame, self._pending_offset, real_sample_count=self.pending_samples.size)
        self.pending_samples = np.empty(0, dtype=np.float32)
        return [row]

    def snapshot(self) -> dict[str, object]:
        return {"stream_id": "demian-v1-audio-stream", "config": asdict(self.config),
                "pending_samples": self.pending_samples.tolist(), "pending_offset": self._pending_offset,
                "frame_index": self._frame_index, "finalized": self._finalized,
                "runtime": self.runtime.snapshot().to_dict(), "coupler": self.coupler.snapshot()}

    def restore(self, snapshot: dict[str, object]) -> None:
        if snapshot.get("stream_id") != "demian-v1-audio-stream" or snapshot.get("config") != asdict(self.config):
            raise ValueError("audio_stream_config_mismatch")
        try:
            pending_raw = np.asarray(snapshot["pending_samples"])
            if pending_raw.ndim != 1:
                raise ValueError
            pending = pending_raw.astype(np.float32, copy=False)
            offset, frame_index = snapshot["pending_offset"], snapshot["frame_index"]
            finalized = snapshot["finalized"]
            runtime, coupler = snapshot["runtime"], snapshot["coupler"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("audio_stream_snapshot_invalid") from error
        if (
            type(offset) is not int or offset < 0 or type(frame_index) is not int or frame_index < 0
            or type(finalized) is not bool or pending.size >= self.config.frame_size
            or (finalized and pending.size != 0) or not np.all(np.isfinite(pending))
            or not isinstance(runtime, dict) or not isinstance(coupler, dict)
        ):
            raise ValueError("audio_stream_snapshot_invalid")
        # Validate on matching throwaway constructions before any live state changes.
        trial_runtime = DemianV1Runtime(self.runtime.config)
        trial_runtime.restore(runtime)
        trial_coupler = DeterministicAudioCoupler(
            self.coupler.weights.shape[0], projection_seed=self.coupler.projection_seed,
            output_gain=self.coupler.output_gain,
        )
        trial_coupler.restore(coupler)
        self.runtime.restore(runtime)
        self.coupler.restore(coupler)
        self.pending_samples, self._pending_offset, self._frame_index, self._finalized = pending, offset, frame_index, finalized


__all__ = ["AudioStreamConfig", "DeterministicAudioCoupler", "IncrementalAudioProcessor"]
