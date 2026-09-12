from __future__ import annotations

import pytest

from demian_v1 import audio_capture
from demian_v1.audio_capture import SoundDeviceCapture


def test_optional_sounddevice_adapter_does_not_import_hardware_package_until_constructed(monkeypatch) -> None:
    real_import_module = audio_capture.importlib.import_module

    def missing_sounddevice(name: str) -> object:
        if name == "sounddevice":
            raise ImportError("not installed")
        return real_import_module(name)

    monkeypatch.setattr(audio_capture.importlib, "import_module", missing_sounddevice)

    with pytest.raises(RuntimeError, match="sounddevice_unavailable"):
        SoundDeviceCapture(device=3, sample_rate=48_000)


def test_sounddevice_adapter_rejects_invalid_explicit_parameters() -> None:
    with pytest.raises(ValueError, match="audio_capture_config_invalid"):
        SoundDeviceCapture(device=-1, sample_rate=0)
