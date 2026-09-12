"""Timed local microphone probe; requires the optional ``audio`` extra."""

from __future__ import annotations

import argparse
from pathlib import Path

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_capture import SoundDeviceCapture
from demian_v1.audio_stream import AudioStreamConfig, DeterministicAudioCoupler, IncrementalAudioProcessor
from demian_v1.live_audio import LiveAudioSession, run_timed_session


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Timed local mono microphone probe")
    result.add_argument("--list-devices", action="store_true")
    result.add_argument("--device", type=int)
    result.add_argument("--sample-rate", type=int)
    result.add_argument("--duration", type=float)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--frame-size", type=int, default=1_200)
    result.add_argument("--hop-size", type=int, default=480)
    result.add_argument("--capacity-samples", type=int, default=48_000)
    return result


def main() -> int:
    arguments = parser().parse_args()
    if arguments.list_devices:
        print(SoundDeviceCapture.list_devices())
        return 0
    if None in (arguments.device, arguments.sample_rate, arguments.duration, arguments.output_dir):
        parser().error("--device, --sample-rate, --duration, and --output-dir are required unless --list-devices")
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=32, seed=0))
    processor = IncrementalAudioProcessor(
        AudioStreamConfig(sample_rate=arguments.sample_rate, frame_size=arguments.frame_size, hop_size=arguments.hop_size),
        runtime=runtime, coupler=DeterministicAudioCoupler(hidden_size=32, projection_seed=0),
    )
    session = LiveAudioSession(
        processor, SoundDeviceCapture(device=arguments.device, sample_rate=arguments.sample_rate),
        output_dir=arguments.output_dir, capacity_samples=arguments.capacity_samples,
    )
    run_timed_session(session, duration_seconds=arguments.duration)
    print(session.summary)
    return 0 if session.summary["experiment_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
