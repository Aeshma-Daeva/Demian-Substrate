"""Deterministic runtime and capsule boundary for Demian v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from development.demian_v1_gate_state import V1_CHANNELS, DemianV1GateState, V1State

DEMIAN_V1_ID = "demian-v1"


@dataclass(frozen=True)
class DemianV1Config:
    """Stable construction parameters for a Demian v1 runtime."""

    hidden_size: int = 32
    seed: int = 0
    gate_disabled: bool = False
    gate_frozen: bool = False
    binding_start_step: int = 1


@dataclass(frozen=True)
class DemianV1Snapshot:
    """JSON-compatible recurrent state capsule."""

    runtime_id: str
    config: dict[str, Any]
    step_index: int
    channels: dict[str, list[list[float]]]
    model_state: dict[str, list[float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_state(state: V1State) -> dict[str, list[list[float]]]:
    """Serialize all six recurrent channels without losing continuity."""

    return {
        channel: tensor.detach().cpu().tolist()
        for channel, tensor in zip(V1_CHANNELS, state, strict=True)
    }


def deserialize_state(
    payload: dict[str, list[list[float]]],
    *,
    device: torch.device | None = None,
) -> V1State:
    """Restore all six recurrent channels from a capsule payload."""

    target = device or torch.device("cpu")
    missing = [channel for channel in V1_CHANNELS if channel not in payload]
    if missing:
        raise ValueError(f"demian_v1_channels_missing:{','.join(missing)}")
    return tuple(
        torch.tensor(payload[channel], dtype=torch.float32, device=target)
        for channel in V1_CHANNELS
    )  # type: ignore[return-value]


class DemianV1Runtime:
    """Small deterministic wrapper suitable for embedding in another runtime."""

    def __init__(self, config: DemianV1Config | None = None) -> None:
        self.config = config or DemianV1Config()
        if self.config.hidden_size < 2:
            raise ValueError("demian_v1_hidden_size_must_be_at_least_two")
        torch.manual_seed(self.config.seed)
        self.model = DemianV1GateState(
            hidden_size=self.config.hidden_size,
            gate_disabled=self.config.gate_disabled,
            gate_frozen=self.config.gate_frozen,
            binding_start_step=self.config.binding_start_step,
        )
        self.model.eval()
        self.state = self.model.initial_state(1, torch.device("cpu"))
        self.step_index = 0

    def step(self, coupling: torch.Tensor | None = None, *, strength: float = 1.0) -> dict[str, Any]:
        """Advance once, optionally injecting a bounded coupling vector."""

        if coupling is not None:
            vector = coupling.detach().to(dtype=torch.float32, device=torch.device("cpu")).view(1, -1)
            if vector.shape[-1] != self.config.hidden_size:
                raise ValueError("demian_v1_coupling_size_mismatch")
            self.state = self.model.inject_coupling_message(self.state, vector, float(strength))
        with torch.no_grad():
            self.state = self.model.step(self.state)
        self.step_index += 1
        return {
            "runtime_id": DEMIAN_V1_ID,
            "step_index": self.step_index,
            "surface": self.model.state_vector(self.state).view(-1).detach().cpu().tolist(),
            "metrics": dict(self.model.step_aux()),
        }

    def snapshot(self) -> DemianV1Snapshot:
        """Capture model parameters and full recurrent state."""

        return DemianV1Snapshot(
            runtime_id=DEMIAN_V1_ID,
            config=asdict(self.config),
            step_index=self.step_index,
            channels=serialize_state(self.state),
            model_state={
                name: tensor.detach().cpu().reshape(-1).tolist()
                for name, tensor in self.model.state_dict().items()
            },
        )

    def restore(self, snapshot: DemianV1Snapshot | dict[str, Any], *, surface_only: bool = False) -> None:
        """Restore full continuity or an explicit surface-only control."""

        payload = snapshot.to_dict() if isinstance(snapshot, DemianV1Snapshot) else dict(snapshot)
        if payload.get("runtime_id") != DEMIAN_V1_ID:
            raise ValueError("demian_v1_runtime_id_mismatch")
        channels = deserialize_state(dict(payload.get("channels") or {}))
        if surface_only:
            surface = self.model.state_vector(channels).detach()
            channels = (
                surface,
                *(torch.zeros_like(channel) for channel in channels[1:]),
            )
        self.state = channels
        self.step_index = max(0, int(payload.get("step_index") or 0))
        self.model._step_index = self.step_index
