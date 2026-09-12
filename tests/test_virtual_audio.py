from __future__ import annotations

import wave

import numpy as np
import pytest

from demian_v1.virtual_audio import WavCaptureBackend


def _write_wav(path, samples: np.ndarray, rate: int = 1_000) -> None:  # type: ignore[no-untyped-def]
    pcm = np.clip(samples, -1, 1)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((pcm * 32767).astype("<i2").tobytes())


def test_wav_capture_pumps_seeded_partitions_through_callback_and_eof(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "tone.wav"
    _write_wav(path, np.arange(10, dtype=np.float32) / 10)
    blocks: list[np.ndarray] = []
    faults: list[str] = []
    capture = WavCaptureBackend(path, callback_sizes=(3, 1, 4), paced=False)

    capture.start(blocks.append, faults.append)
    while capture.pump():
        pass

    assert [block.size for block in blocks] == [3, 1, 4, 2]
    assert np.allclose(np.concatenate(blocks), np.arange(10, dtype=np.float32) / 10, atol=1e-4)
    assert faults == []
    assert capture.eof is True
    assert capture.pump() is False
    assert capture._source is None  # type: ignore[attr-defined]


def test_wav_capture_rejects_invalid_schedule_and_latches_callback_fault(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "silence.wav"
    _write_wav(path, np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="wav_capture_schedule_invalid"):
        WavCaptureBackend(path, callback_sizes=(0,))

    capture = WavCaptureBackend(path, callback_sizes=(2,))
    faults: list[str] = []
    capture.start(lambda _: (_ for _ in ()).throw(RuntimeError("callback_broke")), faults.append)

    assert capture.pump() is False
    assert faults == ["wav_callback_failure:callback_broke"]
    assert capture._source is None  # type: ignore[attr-defined]


def test_wav_capture_construction_does_not_read_or_retain_the_full_pcm_payload(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "long.wav"
    _write_wav(path, np.linspace(-1, 1, 100_000, dtype=np.float32))
    original = wave.Wave_read.readframes

    def fail_if_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("payload_read_during_construction")

    monkeypatch.setattr(wave.Wave_read, "readframes", fail_if_read)
    capture = WavCaptureBackend(path, callback_sizes=(17, 31))

    assert capture.buffered_samples == 0
    monkeypatch.setattr(wave.Wave_read, "readframes", original)
    blocks: list[np.ndarray] = []
    capture.start(blocks.append, lambda _: None)
    assert capture.pump() is True
    assert blocks[0].size == 17
    assert capture.buffered_samples <= 31
