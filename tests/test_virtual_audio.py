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
