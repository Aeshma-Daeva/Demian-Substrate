"""Offline WAV-to-Demian probe with deterministic JSONL traces."""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Iterator

import numpy as np
from demian_v1.audio_stream import (
    AudioStreamConfig,
    DeterministicAudioCoupler,
    IncrementalAudioProcessor,
)
from demian_v1.runtime import DemianV1Config, DemianV1Runtime


def decode_pcm_payload(payload: bytes, *, channels: int, width: int) -> np.ndarray:
    """Decode a byte-aligned integer PCM payload and mix it to mono."""
    if channels < 1 or width not in (1, 2, 3, 4):
        raise ValueError("audio_wav_format_unsupported")
    if width == 1:
        decoded = (np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        decoded = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        decoded = np.frombuffer(payload, dtype="<i4").astype(np.float64) / 2147483648.0
    else:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        integers = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        integers = np.where(integers & 0x800000, integers - 0x1000000, integers)
        decoded = integers.astype(np.float64) / 8388608.0
    if decoded.size % channels:
        raise ValueError("audio_wav_payload_misaligned")
    mono = decoded.reshape(-1, channels).mean(axis=1)
    return mono.astype(np.float32)


def read_pcm_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read integer PCM WAV and mix channels to mono."""

    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE":
            raise ValueError("audio_wav_must_be_uncompressed_pcm")
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
        payload = source.readframes(frames)
    if sample_rate <= 0:
        raise ValueError("audio_wav_format_unsupported")
    return sample_rate, decode_pcm_payload(payload, channels=channels, width=width)


def iter_audio_frames(samples: np.ndarray, frame_size: int, hop_size: int) -> Iterator[tuple[int, np.ndarray]]:
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("audio_frame_and_hop_must_be_positive")
    for offset in range(0, samples.size, hop_size):
        frame = samples[offset : offset + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        yield offset, frame


def run_audio_probe(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    hidden_size: int = 32,
    runtime_seed: int = 0,
    projection_seed: int = 0,
    strength: float = 0.15,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    runtime: DemianV1Runtime | None = None,
    coupler: DeterministicAudioCoupler | None = None,
    frame_index_start: int = 0,
    sample_offset_start: int = 0,
) -> Path:
    """Process a complete WAV file and write one stable JSON object per frame."""

    input_file = Path(input_path)
    output_file = Path(output_path) if output_path is not None else input_file.with_suffix(".demian-voice.jsonl")
    sample_rate, samples = read_pcm_wav(input_file)
    frame_size = max(1, round(sample_rate * frame_ms / 1_000.0))
    hop_size = max(1, round(sample_rate * hop_ms / 1_000.0))
    active_runtime = runtime or DemianV1Runtime(DemianV1Config(hidden_size=hidden_size, seed=runtime_seed))
    active_coupler = coupler or DeterministicAudioCoupler(
        active_runtime.config.hidden_size,
        projection_seed=projection_seed,
    )
    if active_runtime.config.hidden_size != active_coupler.weights.shape[0]:
        raise ValueError("audio_probe_hidden_size_mismatch")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    processor = IncrementalAudioProcessor(
        AudioStreamConfig(
            sample_rate, frame_size, hop_size, strength, "offline", frame_index_start, sample_offset_start
        ),
        runtime=active_runtime, coupler=active_coupler,
    )
    with output_file.open("w", encoding="utf-8", newline="\n") as trace:
        for row in processor.feed(samples) + processor.finalize():
            trace.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return output_file


__all__ = [
    "DeterministicAudioCoupler",
    "decode_pcm_payload",
    "iter_audio_frames",
    "read_pcm_wav",
    "run_audio_probe",
]
