"""Tests for the stable demian-v1 package boundary."""

from __future__ import annotations

import torch

from demian_v1 import DEMIAN_V1_ID, DemianV1Config, DemianV1Runtime, deserialize_state, serialize_state


def test_public_runtime_uses_stable_identity_and_six_channel_capsules() -> None:
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=101))

    result = runtime.step(torch.ones(8) * 0.05)
    snapshot = runtime.snapshot()

    assert result["runtime_id"] == DEMIAN_V1_ID == "demian-v1"
    assert tuple(snapshot.channels) == ("fast", "slow", "control", "message", "carrier", "gate")
    assert snapshot.step_index == 1


def test_state_serialization_round_trip_preserves_every_channel() -> None:
    runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=102))
    runtime.step()

    restored = deserialize_state(serialize_state(runtime.state))

    assert all(torch.equal(left, right) for left, right in zip(runtime.state, restored, strict=True))


def test_full_restore_is_deterministic_and_surface_control_diverges() -> None:
    source = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=103))
    source.step(torch.linspace(-0.1, 0.1, 8))
    snapshot = source.snapshot()

    full = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=103))
    full.restore(snapshot)
    surface = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=103))
    surface.restore(snapshot, surface_only=True)

    expected = source.step()["surface"]
    actual = full.step()["surface"]
    control = surface.step()["surface"]

    assert torch.allclose(torch.tensor(expected), torch.tensor(actual))
    assert not torch.allclose(torch.tensor(expected), torch.tensor(control))
