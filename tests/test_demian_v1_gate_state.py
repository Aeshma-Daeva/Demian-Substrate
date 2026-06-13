"""Tests for the Demian v1 explicit gate-state substrate."""

from __future__ import annotations

import torch

from development.demian_v1_gate_state import (
    V1_CHANNELS,
    DemianV1GateState,
    clamp_v1_channel,
    compare_v1_resume,
    run_v1_trace,
    surface_only_resume_state,
)


def test_v1_state_has_explicit_gate_channel() -> None:
    torch.manual_seed(94)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))

    assert len(state) == 6
    assert tuple(model.state_components(state)) == V1_CHANNELS
    assert model.state_components(state)["gate"].shape == (1, 8)


def test_surface_only_resume_keeps_surface_and_zeros_internal_capsule() -> None:
    torch.manual_seed(94)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))
    resumed = surface_only_resume_state(model, state)

    assert torch.allclose(resumed[0], model.state_vector(state))
    for channel in resumed[1:]:
        assert torch.count_nonzero(channel) == 0


def test_gate_disabled_clamps_gate_after_trace_step() -> None:
    model = DemianV1GateState(hidden_size=8, gate_disabled=True)

    _, state, metrics = run_v1_trace(model, steps=4, seed=95)

    assert torch.count_nonzero(state[-1]) == 0
    assert metrics[-1]["gate_disabled"] == 1.0


def test_gate_modulates_routes_without_release_vector_metrics() -> None:
    torch.manual_seed(96)
    model = DemianV1GateState(
        hidden_size=8,
        gate_to_message_carrier_scale=0.9,
        gate_to_carrier_slow_scale=0.8,
        gate_to_surface_scale=0.7,
    )
    state = model.initial_state(1, torch.device("cpu"))

    _ = model.step(state)
    aux = model.step_aux()

    assert aux["gate_state_norm"] > 0.0
    assert aux["gate_change_duty"] >= 0.0
    assert aux["gate_to_message_carrier_scale"] != 1.0
    assert aux["gate_to_carrier_slow_scale"] != 1.0
    assert aux["gate_to_surface_scale"] != 1.0
    assert "release_bias_norm" not in aux
    assert "release_open_mean" not in aux
    assert "endogenous_release_norm" not in aux


def test_channel_clamp_validates_known_v1_channels() -> None:
    torch.manual_seed(97)
    model = DemianV1GateState(hidden_size=8)
    state = model.initial_state(1, torch.device("cpu"))

    clamped = clamp_v1_channel(state, "message")

    assert torch.count_nonzero(clamped[V1_CHANNELS.index("message")]) == 0


def test_full_capsule_resume_outperforms_surface_only_resume() -> None:
    model = DemianV1GateState(hidden_size=8)

    full = compare_v1_resume(model, seed=98, pause_steps=6, resume_steps=6, surface_only=False)
    surface = compare_v1_resume(model, seed=98, pause_steps=6, resume_steps=6, surface_only=True)

    assert full.mean_step_gap < 1e-7
    assert surface.mean_step_gap > full.mean_step_gap
