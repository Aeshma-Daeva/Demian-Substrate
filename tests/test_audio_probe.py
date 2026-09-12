from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_probe import DeterministicAudioCoupler, read_pcm_wav, run_audio_probe


def _pcm_payload(width: int, values: list[float]) -> bytes:
    if width == 1:
        return bytes(round((value + 1.0) * 127.5) for value in values)
    if width == 2:
        return struct.pack(f"<{len(values)}h", *(round(value * 32767) for value in values))
    if width == 3:
        payload = bytearray()
        for value in values:
            integer = round(value * 8388607)
            payload.extend((integer & 0xFFFFFF).to_bytes(3, "little"))
        return bytes(payload)
    return struct.pack(f"<{len(values)}i", *(round(value * 2147483647) for value in values))


def _write_wav(path: Path, *, width: int = 2, channels: int = 1, frames: int = 400) -> None:
    values: list[float] = []
    for index in range(frames):
        sample = 0.6 * math.sin(2.0 * math.pi * 120.0 * index / 8_000)
        values.extend([sample, -sample] if channels == 2 else [sample])
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(8_000)
        wav.writeframes(_pcm_payload(width, values))


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_pcm_wav_reader_normalizes_supported_integer_widths(tmp_path: Path, width: int) -> None:
    path = tmp_path / f"width-{width}.wav"
    _write_wav(path, width=width)

    sample_rate, samples = read_pcm_wav(path)

    assert sample_rate == 8_000
    assert samples.dtype == np.float32
    assert samples.shape == (400,)
    assert float(np.max(np.abs(samples))) == pytest.approx(0.6, abs=0.02)


def test_multichannel_wav_is_mixed_to_mono(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    _write_wav(path, channels=2)

    _, samples = read_pcm_wav(path)

    assert np.max(np.abs(samples)) < 1e-4


def test_probe_writes_observable_deterministic_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "voice.wav"
    first_trace = tmp_path / "first.jsonl"
    second_trace = tmp_path / "second.jsonl"
    _write_wav(path, frames=401)

    run_audio_probe(path, first_trace, hidden_size=8, runtime_seed=5, projection_seed=7)
    run_audio_probe(path, second_trace, hidden_size=8, runtime_seed=5, projection_seed=7)

    assert first_trace.read_bytes() == second_trace.read_bytes()
    rows = [json.loads(line) for line in first_trace.read_text().splitlines()]
    assert [row["sample_offset"] for row in rows] == [0, 80, 160, 240, 320, 400]
    assert rows[0]["time_seconds"] == 0.0
    assert len(rows[0]["features"]) == 13
    assert len(rows[0]["coupling"]) == 8
    assert set(rows[0]["channel_norms"]) == {"fast", "slow", "control", "message", "carrier", "gate"}
    assert rows[-1]["frame_index"] == 5


def test_probe_checkpoint_continuation_preserves_global_indices(tmp_path: Path) -> None:
    path = tmp_path / "voice.wav"
    trace = tmp_path / "continued.jsonl"
    _write_wav(path, frames=240)
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=9))
    coupler = DeterministicAudioCoupler(hidden_size=8, projection_seed=11)

    run_audio_probe(
        path,
        trace,
        runtime=runtime,
        coupler=coupler,
        frame_index_start=100,
        sample_offset_start=1_600,
    )

    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert rows[0]["frame_index"] == 100
    assert rows[0]["sample_offset"] == 1_600
    assert rows[0]["time_seconds"] == pytest.approx(0.2)


def test_cli_uses_default_trace_name(tmp_path: Path) -> None:
    from development.probe_v1_voice_dynamics import main

    path = tmp_path / "voice.wav"
    _write_wav(path, frames=80)

    assert main([str(path), "--hidden-size", "8"]) == 0
    assert path.with_suffix(".demian-voice.jsonl").is_file()


def test_audio_probe_is_available_from_the_stable_package_boundary() -> None:
    from demian_v1 import AudioFeatureExtractor, DeterministicAudioCoupler, run_audio_probe

    assert AudioFeatureExtractor.__name__ == "AudioFeatureExtractor"
    assert DeterministicAudioCoupler.__name__ == "DeterministicAudioCoupler"
    assert callable(run_audio_probe)
