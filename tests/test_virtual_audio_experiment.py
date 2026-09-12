from __future__ import annotations

import json

import numpy as np

from development.virtual_audio_experiment import controlled_signal, run_virtual_experiment, write_pcm_wav


def test_controlled_signals_are_deterministic_and_nonsemantic() -> None:
    names = ("silence", "tone120", "tone300", "quiet_loud", "sweep", "noise_burst", "rhythm")
    first = {name: controlled_signal(name, sample_rate=1_000, duration_seconds=0.2) for name in names}
    second = {name: controlled_signal(name, sample_rate=1_000, duration_seconds=0.2) for name in names}

    assert all(samples.dtype == np.float32 and samples.size == 200 for samples in first.values())
    assert np.array_equal(first["noise_burst"], second["noise_burst"])
    assert np.max(np.abs(first["silence"])) == 0
    assert np.max(np.abs(first["quiet_loud"][:100])) < np.max(np.abs(first["quiet_loud"][100:]))


def test_virtual_experiment_writes_derived_trace_only(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.wav"
    write_pcm_wav(source, 1_000, controlled_signal("rhythm", sample_rate=1_000, duration_seconds=0.08))

    result = run_virtual_experiment(source, tmp_path / "out", callback_sizes=(3, 5, 2), frame_ms=4, hop_ms=4)

    assert result["state"] == "STOPPED"
    assert (tmp_path / "out" / "frames.jsonl").is_file()
    assert (tmp_path / "out" / "events.jsonl").is_file()
    assert not list((tmp_path / "out").glob("*.wav"))
    rows = [json.loads(line) for line in (tmp_path / "out" / "frames.jsonl").read_text().splitlines()]
    assert rows and all("features" in row and "surface" in row for row in rows)


def test_many_seeded_callback_partitions_preserve_a_complete_accelerated_prefix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source.wav"
    samples = controlled_signal("sweep", sample_rate=1_000, duration_seconds=2.0)
    write_pcm_wav(source, 1_000, samples)
    for seed in range(8):
        schedule = tuple(np.random.default_rng(seed).integers(1, 90, size=5).tolist())
        result = run_virtual_experiment(source, tmp_path / f"run-{seed}", callback_sizes=schedule, frame_ms=20, hop_ms=10)
        assert result["experiment_valid"] is True
        assert result["accepted_samples"] == samples.size
        assert result["peak_occupancy_samples"] <= max(schedule)
