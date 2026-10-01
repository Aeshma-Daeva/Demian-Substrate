"""Focused tests for the accumulating fixed-point baseline protocol."""

from __future__ import annotations

from development.afp_baselines import AFPConfig, match_baseline_hidden_size, run_experiment


def test_parameter_matching_returns_finite_baseline_sizes() -> None:
    target = 10_000
    for kind in ("rnn", "gru", "lstm"):
        hidden_size, parameter_count = match_baseline_hidden_size(
            kind,
            target_parameters=target,
            input_size=8,
            seed=94,
            max_hidden=128,
        )
        assert hidden_size >= 2
        assert parameter_count > 0
        assert abs(parameter_count - target) / target < 0.35


def test_small_afp_protocol_is_deterministic_and_complete() -> None:
    config = AFPConfig(
        demian_hidden_size=4,
        readout_dim=2,
        histories=4,
        drive_steps=6,
        shared_tail_steps=2,
        settle_steps=4,
        analysis_window=2,
        probe_steps=3,
        input_scale=0.10,
        probe_scale=0.10,
        max_baseline_hidden=64,
    )

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second
    assert tuple(first["architectures"]) == ("rnn", "gru", "lstm", "demian_v1")

    for payload in first["architectures"].values():
        assert payload["parameter_count"] > 0
        assert payload["native_surface_dim"] >= config.readout_dim
        assert payload["latent_dim"] >= payload["native_surface_dim"]
        assert len(payload["histories"]) == config.histories
        assert 0.0 <= payload["aggregate"]["native_afp_fraction"] <= 1.0
        assert 0.0 <= payload["aggregate"]["projected_afp_fraction"] <= 1.0
        assert "selected_hidden_collision_pair" in payload["pair_search"]
        assert len(payload["continuation"]["shared_probe_future"]["gaps"]) == config.probe_steps
