"""Offline file-backed capture for exercising the live callback boundary."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import numpy as np

from demian_v1.audio_probe import read_pcm_wav


class WavCaptureBackend:
    """Emit local PCM WAV samples via capture callbacks in deterministic chunks.

    ``pump`` does one callback and never sleeps. ``run`` optionally supplies
    real-time pacing outside callback execution; accelerated tests use pump.
    """

    def __init__(self, source: str | Path, *, callback_sizes: tuple[int, ...] = (256,),
                 paced: bool = False) -> None:
        if not callback_sizes or any(type(size) is not int or size <= 0 for size in callback_sizes):
            raise ValueError("wav_capture_schedule_invalid")
        self.sample_rate, self._samples = read_pcm_wav(source)
        self.callback_sizes, self.paced = callback_sizes, paced
        self._offset = 0
        self._schedule_index = 0
        self._on_block: Callable[[np.ndarray], None] | None = None
        self._on_fault: Callable[[str], None] | None = None
        self._started = False
        self._stopped = False
        self.eof = False

    def start(self, on_block: Callable[[np.ndarray], None], on_fault: Callable[[str], None]) -> None:
        if self._started:
            raise ValueError("wav_capture_already_started")
        self._on_block, self._on_fault, self._started = on_block, on_fault, True

    def stop(self) -> None:
        self._stopped = True

    def _fault(self, detail: str) -> None:
        self._stopped = True
        if self._on_fault is not None:
            self._on_fault(detail)

    def pump(self) -> bool:
        """Deliver one block, returning false at EOF, stop, or callback fault."""
        if not self._started:
            raise ValueError("wav_capture_not_started")
        if self._stopped or self.eof:
            return False
        if self._offset >= self._samples.size:
            self.eof = True
            return False
        size = self.callback_sizes[self._schedule_index % len(self.callback_sizes)]
        self._schedule_index += 1
        next_offset = min(self._offset + size, self._samples.size)
        block = self._samples[self._offset:next_offset].copy()
        self._offset = next_offset
        try:
            assert self._on_block is not None
            self._on_block(block)
        except Exception as error:
            self._fault(f"wav_callback_failure:{error}")
            return False
        if self._offset >= self._samples.size:
            self.eof = True
        return True

    def run(self, *, sleep: Callable[[float], None] = time.sleep) -> None:
        """Consume to EOF; pacing is deliberately outside the callback."""
        while not self._stopped and not self.eof:
            before = self._offset
            if not self.pump():
                break
            if self.paced:
                sleep((self._offset - before) / self.sample_rate)


__all__ = ["WavCaptureBackend"]
