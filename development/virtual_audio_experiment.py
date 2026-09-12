"""Run controlled local WAV material through the live-audio callback boundary."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np

from demian_v1 import DemianV1Config, DemianV1Runtime
from demian_v1.audio_stream import AudioStreamConfig, DeterministicAudioCoupler, IncrementalAudioProcessor
from demian_v1.live_audio import LiveAudioSession
from demian_v1.virtual_audio import WavCaptureBackend


SIGNALS = ("silence", "tone120", "tone300", "quiet_loud", "sweep", "noise_burst", "rhythm")


def controlled_signal(name: str, *, sample_rate: int = 48_000, duration_seconds: float = 1.0) -> np.ndarray:
    """Produce deterministic non-semantic test material; no downloaded or spoken audio."""
    if name not in SIGNALS or sample_rate <= 0 or duration_seconds <= 0:
        raise ValueError("controlled_signal_config_invalid")
    count = round(sample_rate * duration_seconds)
    time = np.arange(count, dtype=np.float64) / sample_rate
    if name == "silence":
        values = np.zeros(count)
    elif name == "tone120":
        values = 0.5 * np.sin(2 * np.pi * 120 * time)
    elif name == "tone300":
        values = 0.5 * np.sin(2 * np.pi * 300 * time)
    elif name == "quiet_loud":
        gain = np.where(time < duration_seconds / 2, 0.05, 0.7)
        values = gain * np.sin(2 * np.pi * 180 * time)
    elif name == "sweep":
        phase = 2 * np.pi * (80 * time + 0.5 * (900 - 80) / duration_seconds * time**2)
        values = 0.5 * np.sin(phase)
    elif name == "noise_burst":
        rng = np.random.default_rng(8_731)
        gate = (time >= duration_seconds * 0.4) & (time < duration_seconds * 0.6)
        values = gate * rng.normal(0, 0.35, count)
    else:
        beat = (np.mod(time, 0.16) < 0.025).astype(float)
        values = (0.26 * np.sin(2 * np.pi * 180 * time) + 0.14 * np.sin(2 * np.pi * 360 * time)) * (0.3 + 0.7 * beat)
    return values.astype(np.float32)


def write_pcm_wav(path: str | Path, sample_rate: int, samples: np.ndarray) -> Path:
    """Write a development-only mono PCM fixture."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return output


def run_virtual_experiment(input_wav: str | Path, output_dir: str | Path, *, callback_sizes: tuple[int, ...] = (127, 251, 509),
                           frame_ms: float = 25.0, hop_ms: float = 10.0) -> dict[str, object]:
    """Accelerate a local WAV through ``LiveAudioSession`` and retain only traces."""
    source = WavCaptureBackend(input_wav, callback_sizes=callback_sizes, paced=False)
    frame_size, hop_size = max(1, round(source.sample_rate * frame_ms / 1_000)), max(1, round(source.sample_rate * hop_ms / 1_000))
    processor = IncrementalAudioProcessor(
        AudioStreamConfig(source.sample_rate, frame_size, hop_size, segment_id="virtual"),
        runtime=DemianV1Runtime(DemianV1Config(hidden_size=8, seed=4)),
        coupler=DeterministicAudioCoupler(8, projection_seed=6),
    )
    session = LiveAudioSession(processor, source, output_dir=Path(output_dir), capacity_samples=max(callback_sizes) * 2)
    session.start()
    while source.pump():
        session.drain()
        if session.state.value == "FAILED":
            break
    session.drain()
    session.stop()
    return session.summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-wav", type=Path)
    source.add_argument("--signal", choices=SIGNALS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    args = parser.parse_args(argv)
    input_wav = args.input_wav
    if input_wav is None:
        input_wav = args.output_dir / ".fixture.wav"
        write_pcm_wav(input_wav, args.sample_rate, controlled_signal(args.signal, sample_rate=args.sample_rate, duration_seconds=args.duration))
    run_virtual_experiment(input_wav, args.output_dir)
    if args.signal is not None:
        input_wav.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
