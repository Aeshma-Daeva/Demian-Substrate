from __future__ import annotations

import builtins

import pytest

from demian_v1.audio_capture import SoundDeviceCapture


def test_optional_sounddevice_adapter_does_not_import_hardware_package_until_constructed(monkeypatch) -> None:
    real_import = builtins.__import__

    def missing_sounddevice(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sounddevice)

    with pytest.raises(RuntimeError, match="sounddevice_unavailable"):
        SoundDeviceCapture(device=3, sample_rate=48_000)


def test_sounddevice_adapter_rejects_invalid_explicit_parameters() -> None:
    with pytest.raises(ValueError, match="audio_capture_config_invalid"):
        SoundDeviceCapture(device=-1, sample_rate=0)
