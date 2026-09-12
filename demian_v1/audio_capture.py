"""Optional sounddevice capture adapter for the local live-audio boundary."""

from __future__ import annotations

from collections.abc import Callable
import importlib

import numpy as np


class SoundDeviceCapture:
    """Explicit mono float32 device capture with no fallback, resampling, or reconnect."""

    def __init__(self, *, device: int, sample_rate: int, block_size: int = 0) -> None:
        if device < 0 or sample_rate <= 0 or block_size < 0:
            raise ValueError("audio_capture_config_invalid")
        try:
            self._sd = importlib.import_module("sounddevice")
        except ImportError as error:
            raise RuntimeError("sounddevice_unavailable") from error
        self.device, self.sample_rate, self.block_size = device, sample_rate, block_size
        self._stream = None

    @classmethod
    def list_devices(cls) -> object:
        try:
            return importlib.import_module("sounddevice").query_devices()
        except ImportError as error:
            raise RuntimeError("sounddevice_unavailable") from error

    def start(self, on_block: Callable[[np.ndarray], None], on_fault: Callable[[str], None]) -> None:
        try:
            self._sd.check_input_settings(device=self.device, samplerate=self.sample_rate, channels=1, dtype="float32")
            def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
                del frames, time_info
                if status:
                    on_fault(f"backend_status:{status}")
                    return
                on_block(np.asarray(indata[:, 0], dtype=np.float32).copy())
            self._stream = self._sd.InputStream(
                device=self.device, samplerate=self.sample_rate, channels=1, dtype="float32",
                blocksize=self.block_size, callback=callback,
            )
            self._stream.start()
        except Exception as error:
            on_fault(f"device_start_failure:{error}")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


__all__ = ["SoundDeviceCapture"]
