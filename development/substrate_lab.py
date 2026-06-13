"""Self-loop substrate test harness.

This module is intentionally separate from the main Demian runtime.
It provides a small, uniform empirical battery for recurrent substrates:

- vanilla RNN
- GRU
- LSTM
- diagonal SSM
- selective SSM

The goal is not downstream task performance. The goal is to measure:

- rest-state discovery
- basin structure
- perturbation recovery
- memory persistence

All substrates run under the same self-only loop:
    h_t = F(h_{t-1})

When a substrate requires an explicit input channel, a small projection of the
previous state is fed back as the current input.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from legacy.demian_runtime.machine_observables import classify_attractor, compute_observables


def _fft_spectrum(residual: np.ndarray) -> tuple[float, float]:
    x = residual - residual.mean()
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x))
    total = power.sum()
    if total < 1e-20:
        return 0.0, 0.0
    centroid = float(np.sum(freqs * power) / total)
    mf = freqs.max() if freqs.max() > 0 else 1.0
    cn = min(centroid / mf, 1.0)
    sp = np.sort(power)[::-1]
    tn = max(1, len(sp) // 20)
    conc = float(sp[:tn].sum() / total)
    return cn, conc


@dataclass
class StepMetrics:
    step: int
    residual_norm: float
    residual_delta: float
    temporal_coherence: float
    velocity_align: float
    spectral_centroid: float
    spectral_concentration: float
    layer_work_ratio: float
    fast_state_norm: float = 0.0
    slow_state_norm: float = 0.0
    message_state_norm: float = 0.0
    fast_state_delta: float = 0.0
    slow_state_delta: float = 0.0
    message_state_delta: float = 0.0
    fast_contraction_ratio: float = 1.0
    slow_contraction_ratio: float = 1.0
    message_contraction_ratio: float = 1.0
    slow_write_mean: float = 0.0
    message_write_mean: float = 0.0
    fast_update_mean: float = 0.0
    route_metrics: Optional[Dict[str, float]] = None


@dataclass
class RunSummary:
    substrate: str
    seed: int
    steps: int
    perturb_step: Optional[int]
    perturb_scale: float
    final_norm: float
    mean_norm: float
    std_norm: float
    mean_delta: float
    mean_coherence: float
    mean_velocity_align: float
    cycle_period: float
    two_cycle_amplitude: float
    covariance_rank: float
    flow_dimension: float
    compression_ratio: float
    attractor_type: str
    interior_class: str
    attractor_confidence: float
    final_state_checksum: float
    mean_fast_norm: float = 0.0
    mean_slow_norm: float = 0.0
    mean_message_norm: float = 0.0
    max_fast_norm: float = 0.0
    max_slow_norm: float = 0.0
    max_message_norm: float = 0.0
    mean_fast_delta: float = 0.0
    mean_slow_delta: float = 0.0
    mean_message_delta: float = 0.0
    mean_fast_contraction: float = 1.0
    mean_slow_contraction: float = 1.0
    mean_message_contraction: float = 1.0
    max_fast_contraction: float = 1.0
    max_slow_contraction: float = 1.0
    max_message_contraction: float = 1.0
    slow_fast_delta_ratio: float = 0.0
    mean_slow_write: float = 0.0
    mean_message_write: float = 0.0
    mean_fast_update: float = 0.0


def _fixed_projection(in_dim: int, out_dim: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    mat = torch.randn(out_dim, in_dim, generator=gen)
    mat = mat / (mat.norm(dim=1, keepdim=True) + 1e-10)
    return mat


def _encode_state(
    h: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    z = proj @ h
    return torch.sign(z).clamp(min=-1.0, max=1.0)


def _decode_code(
    code: torch.Tensor,
    proj: torch.Tensor,
) -> torch.Tensor:
    recon = proj.T @ code
    rn = recon.norm()
    if rn > 1e-10:
        recon = recon / rn
    return recon


def _code_stats(codes: List[torch.Tensor]) -> Dict[str, float]:
    if not codes:
        return {"unique_codes": 0, "reuse_ratio": 0.0, "code_entropy": 0.0}
    keys = [tuple(int(v.item()) for v in code) for code in codes]
    counts: Dict[tuple[int, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    probs = np.array(list(counts.values()), dtype=np.float64)
    probs = probs / probs.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "unique_codes": len(counts),
        "reuse_ratio": 1.0 - (len(counts) / len(codes)),
        "code_entropy": entropy,
    }


class SelfLoopSubstrate(nn.Module):
    """Abstract recurrent substrate with a self-input channel."""

    def __init__(
        self,
        hidden_size: int,
        init_scale: float = 0.2,
        feedback_scale: float = 1.0,
        state_gain: float = 1.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.init_scale = init_scale
        self.feedback_scale = feedback_scale
        self.state_gain = state_gain
        self._step_aux: Dict[str, float] = {}

    def initial_state(self, batch_size: int, device: torch.device) -> object:
        raise NotImplementedError

    def state_vector(self, state: object) -> torch.Tensor:
        raise NotImplementedError

    def step(self, state: object) -> object:
        raise NotImplementedError

    def state_components(self, state: object) -> Dict[str, torch.Tensor]:
        return {"fast": self.state_vector(state)}

    def step_aux(self) -> Dict[str, float]:
        return dict(self._step_aux)

    def write_surface_state(self, state: object, surface: torch.Tensor) -> object:
        surface = surface.view(1, -1)
        if isinstance(state, tuple):
            return (surface, *state[1:])
        return surface

    def inject_coupling_message(self, state: object, message: torch.Tensor, strength: float) -> object:
        delta = strength * message.view(1, -1)
        if isinstance(state, tuple):
            return (state[0] + delta, *state[1:])
        return state + delta


class VanillaRNNSubstrate(SelfLoopSubstrate):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.cell = nn.RNNCell(hidden_size, hidden_size, nonlinearity="tanh")

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale

    def state_vector(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feedback_scale * self.in_proj(state)
        return self.state_gain * self.cell(x, state)


class GRUSubstrate(SelfLoopSubstrate):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.cell = nn.GRUCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale

    def state_vector(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feedback_scale * self.in_proj(state)
        return self.state_gain * self.cell(x, state)


class DualGRUSubstrate(SelfLoopSubstrate):
    """Two-timescale GRU.

    The fast state is the exposed, rapidly updated surface.
    The slow state is the continuity carrier, updated more conservatively.
    The two states are weakly coupled so the slow state can anchor the fast
    state without fully freezing it.
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.fast_in = nn.Linear(hidden_size * 2, hidden_size)
        self.slow_in = nn.Linear(hidden_size * 2, hidden_size)
        self.fast_cell = nn.GRUCell(hidden_size, hidden_size)
        self.slow_cell = nn.GRUCell(hidden_size, hidden_size)
        self.fast_to_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slow_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.expose_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.slow_mix = 0.15
        self.fast_mix = 0.35

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        return fast, slow

    def state_vector(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fast, slow = state
        gate = torch.sigmoid(self.expose_gate(torch.cat([fast, slow], dim=-1)))
        return gate * fast + (1.0 - gate) * slow

    def step(self, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        fast, slow = state
        stacked = torch.cat([fast, slow], dim=-1)
        fast_x = self.feedback_scale * self.fast_in(stacked)
        slow_x = self.feedback_scale * self.slow_in(stacked)

        fast_candidate = self.fast_cell(fast_x, fast)
        slow_candidate = self.slow_cell(slow_x, slow)

        coupled_fast = fast_candidate + self.fast_mix * torch.tanh(self.slow_to_fast(slow))
        coupled_slow = slow_candidate + self.slow_mix * torch.tanh(self.fast_to_slow(fast))

        new_fast = self.state_gain * coupled_fast
        new_slow = coupled_slow
        return new_fast, new_slow


class DualGRUV2Substrate(SelfLoopSubstrate):
    """Asymmetric two-timescale GRU.

    Slow state should act more like continuity anchor than peer.
    Fast state should read heavily from slow state.
    Slow state should only admit filtered, weak writes from fast state.
    """

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.fast_in = nn.Linear(hidden_size * 2, hidden_size)
        self.slow_in = nn.Linear(hidden_size * 2, hidden_size)
        self.fast_cell = nn.GRUCell(hidden_size, hidden_size)
        self.slow_cell = nn.GRUCell(hidden_size, hidden_size)
        self.fast_to_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.slow_to_fast = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fast_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.slow_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.expose_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.fast_mix = 0.75
        self.slow_mix = 0.08
        self.slow_persistence = 0.92

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        return fast, slow

    def state_vector(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fast, slow = state
        gate = torch.sigmoid(self.expose_gate(torch.cat([fast, slow], dim=-1)))
        return gate * fast + (1.0 - gate) * slow

    def step(self, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        fast, slow = state
        stacked = torch.cat([fast, slow], dim=-1)
        fast_x = self.feedback_scale * self.fast_in(stacked)
        slow_x = self.feedback_scale * self.slow_in(stacked)

        fast_candidate = self.fast_cell(fast_x, fast)
        slow_candidate = self.slow_cell(slow_x, slow)

        slow_to_fast_gate = torch.sigmoid(self.fast_gate(stacked))
        fast_to_slow_gate = torch.sigmoid(self.slow_gate(stacked))

        anchored_fast = fast_candidate + self.fast_mix * slow_to_fast_gate * torch.tanh(self.slow_to_fast(slow))
        filtered_write = self.slow_mix * fast_to_slow_gate * torch.tanh(self.fast_to_slow(fast_candidate))
        anchored_slow = self.slow_persistence * slow + (1.0 - self.slow_persistence) * slow_candidate + filtered_write

        new_fast = self.state_gain * anchored_fast
        new_slow = anchored_slow
        return new_fast, new_slow


class DualGRUV3Substrate(SelfLoopSubstrate):
    """Asymmetric dual-GRU with persistent message state.

    Fast state explores local orbit geometry.
    Slow state has its own learned recurrence and conservative write gate.
    Message state is low-dimensional, persistent, and used as the coupling surface.
    """

    def __init__(
        self,
        hidden_size: int,
        slow_size: Optional[int] = None,
        message_dim: Optional[int] = None,
        slow_gate_bias: float = -1.5,
        message_gate_bias: float = -0.75,
        fast_mix: float = 0.45,
        slow_mix: float = 0.18,
        message_mix: float = 0.4,
        slow_step_scale: float = 0.35,
        message_step_scale: float = 0.8,
        fast_update_bias: float = 0.2,
        coupling_to_fast: float = 0.1,
        coupling_to_message: float = 1.0,
        message_init_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.slow_size = slow_size or hidden_size
        self.message_dim = message_dim or max(4, hidden_size // 8)
        self.slow_gate_bias = slow_gate_bias
        self.message_gate_bias = message_gate_bias
        self.fast_mix = fast_mix
        self.slow_mix = slow_mix
        self.message_mix = message_mix
        self.slow_step_scale = slow_step_scale
        self.message_step_scale = message_step_scale
        self.fast_update_bias = fast_update_bias
        self.coupling_to_fast = coupling_to_fast
        self.coupling_to_message = coupling_to_message
        self.message_init_scale = message_init_scale if message_init_scale is not None else self.init_scale

        self.fast_in = nn.Linear(hidden_size + self.slow_size + self.message_dim, hidden_size)
        self.slow_in = nn.Linear(hidden_size + self.message_dim, self.slow_size)
        self.fast_cell = nn.GRUCell(hidden_size, hidden_size)
        self.slow_cell = nn.GRUCell(self.slow_size, self.slow_size)

        self.slow_to_fast = nn.Linear(self.slow_size, hidden_size, bias=False)
        self.message_to_fast = nn.Linear(self.message_dim, hidden_size, bias=False)
        self.fast_to_slow = nn.Linear(hidden_size + self.message_dim, self.slow_size, bias=False)

        self.slow_write_gate = nn.Linear(hidden_size + self.slow_size + self.message_dim, self.slow_size)
        self.message_write_gate = nn.Linear(hidden_size + self.slow_size + self.message_dim, self.message_dim)
        self.fast_update_gate = nn.Linear(hidden_size + self.slow_size + self.message_dim, hidden_size)
        self.message_proj = nn.Linear(hidden_size + self.slow_size, self.message_dim)

        self.slow_readout = nn.Linear(self.slow_size, hidden_size, bias=False)
        self.message_readout = nn.Linear(self.message_dim, hidden_size, bias=False)
        self.expose_gate = nn.Linear(hidden_size * 3, hidden_size)
        self.coupling_proj = nn.Linear(hidden_size, self.message_dim, bias=False)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.slow_size, device=device) * self.init_scale
        message = torch.randn(batch_size, self.message_dim, device=device) * self.message_init_scale
        return fast, slow, message

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, message = state
        return {"fast": fast, "slow": slow, "message": message}

    def state_vector(self, state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        fast, slow, message = state
        slow_view = torch.tanh(self.slow_readout(slow))
        message_view = torch.tanh(self.message_readout(message))
        gate = torch.sigmoid(self.expose_gate(torch.cat([fast, slow_view, message_view], dim=-1)))
        anchored = 0.6 * slow_view + 0.4 * message_view
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, message = state
        stacked = torch.cat([fast, slow, message], dim=-1)
        fast_x = self.feedback_scale * self.fast_in(stacked)
        fast_candidate = self.fast_cell(fast_x, fast)

        slow_gate_in = torch.cat([fast_candidate, slow, message], dim=-1)
        slow_write = torch.sigmoid(self.slow_write_gate(slow_gate_in) + self.slow_gate_bias)
        slow_write = self.slow_step_scale * slow_write
        slow_x = self.feedback_scale * self.slow_in(torch.cat([fast_candidate, message], dim=-1))
        slow_candidate = self.slow_cell(slow_x, slow)
        slow_residual = self.slow_mix * torch.tanh(self.fast_to_slow(torch.cat([fast_candidate, message], dim=-1)))
        new_slow = (1.0 - slow_write) * slow + slow_write * (slow_candidate + slow_residual)

        message_gate_in = torch.cat([fast_candidate, new_slow, message], dim=-1)
        message_write = torch.sigmoid(self.message_write_gate(message_gate_in) + self.message_gate_bias)
        message_write = self.message_step_scale * message_write
        message_candidate = torch.tanh(self.message_proj(torch.cat([fast_candidate, new_slow], dim=-1)))
        new_message = (1.0 - message_write) * message + message_write * message_candidate

        fast_anchor = (
            fast_candidate
            + self.fast_mix * torch.tanh(self.slow_to_fast(new_slow))
            + self.message_mix * torch.tanh(self.message_to_fast(new_message))
        )
        fast_update = torch.sigmoid(self.fast_update_gate(torch.cat([fast_candidate, new_slow, new_message], dim=-1)) + self.fast_update_bias)
        fast_target = self.state_gain * torch.tanh(fast_anchor)
        new_fast = (1.0 - fast_update) * fast + fast_update * fast_target

        self._step_aux = {
            "fast_update_mean": float(fast_update.mean().item()),
            "slow_write_mean": float(slow_write.mean().item()),
            "message_write_mean": float(message_write.mean().item()),
        }
        return new_fast, new_slow, new_message

    def write_surface_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, message = state
        return surface.view(1, -1), slow, message

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, message_state = state
        message_delta = self.coupling_to_message * strength * torch.tanh(self.coupling_proj(message.view(1, -1)))
        fast_delta = self.coupling_to_fast * strength * message.view(1, -1)
        return fast + fast_delta, slow, message_state + message_delta


class DualGRUV3BSubstrate(DualGRUV3Substrate):
    """Interior-enriched fixed-point substrate.

    Keeps the macro basin contractive while making message and slow channels
    less norm-losing and more self-maintaining.
    """

    def __init__(
        self,
        hidden_size: int,
        message_self_retention: float = 0.72,
        slow_residual_mix: float = 0.22,
        slow_carry_scale: float = 0.08,
        message_drive_scale: float = 0.18,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            message_mix=kwargs.pop("message_mix", 0.5),
            message_step_scale=kwargs.pop("message_step_scale", 0.9),
            coupling_to_message=kwargs.pop("coupling_to_message", 1.2),
            coupling_to_fast=kwargs.pop("coupling_to_fast", 0.06),
            fast_update_bias=kwargs.pop("fast_update_bias", 0.1),
            **kwargs,
        )
        self.message_self_retention = message_self_retention
        self.slow_residual_mix = slow_residual_mix
        self.slow_carry_scale = slow_carry_scale
        self.message_drive_scale = message_drive_scale
        self.message_self_proj = nn.Linear(self.message_dim, self.message_dim, bias=False)
        self.slow_message_proj = nn.Linear(self.message_dim, self.slow_size, bias=False)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, message = state
        stacked = torch.cat([fast, slow, message], dim=-1)
        fast_x = self.feedback_scale * self.fast_in(stacked)
        fast_candidate = self.fast_cell(fast_x, fast)

        slow_gate_in = torch.cat([fast_candidate, slow, message], dim=-1)
        slow_write = torch.sigmoid(self.slow_write_gate(slow_gate_in) + self.slow_gate_bias)
        slow_write = self.slow_step_scale * slow_write
        slow_x = self.feedback_scale * self.slow_in(torch.cat([fast_candidate, message], dim=-1))
        slow_candidate = self.slow_cell(slow_x, slow)
        slow_residual = self.slow_residual_mix * torch.tanh(self.fast_to_slow(torch.cat([fast_candidate, message], dim=-1)))
        slow_carry = self.slow_carry_scale * torch.tanh(self.slow_message_proj(message))
        new_slow = (1.0 - slow_write) * slow + slow_write * (slow_candidate + slow_residual) + slow_carry

        message_gate_in = torch.cat([fast_candidate, new_slow, message], dim=-1)
        message_write = torch.sigmoid(self.message_write_gate(message_gate_in) + self.message_gate_bias)
        message_write = self.message_step_scale * message_write
        message_candidate = torch.tanh(self.message_proj(torch.cat([fast_candidate, new_slow], dim=-1)))
        message_residual = self.message_self_retention * torch.tanh(self.message_self_proj(message))
        message_drive = self.message_drive_scale * torch.tanh(self.message_proj(torch.cat([fast, slow], dim=-1)))
        new_message = (1.0 - message_write) * message + message_write * message_candidate + message_residual + message_drive

        fast_anchor = (
            fast_candidate
            + self.fast_mix * torch.tanh(self.slow_to_fast(new_slow))
            + self.message_mix * torch.tanh(self.message_to_fast(new_message))
        )
        fast_update = torch.sigmoid(self.fast_update_gate(torch.cat([fast_candidate, new_slow, new_message], dim=-1)) + self.fast_update_bias)
        fast_target = self.state_gain * torch.tanh(fast_anchor)
        new_fast = (1.0 - fast_update) * fast + fast_update * fast_target

        self._step_aux = {
            "fast_update_mean": float(fast_update.mean().item()),
            "slow_write_mean": float(slow_write.mean().item()),
            "message_write_mean": float(message_write.mean().item()),
        }
        return new_fast, new_slow, new_message


class DualGRUV4Substrate(SelfLoopSubstrate):
    """Four-channel recurrent substrate with explicit carrier and packet paths.

    The previous v3/v3b line overloaded one "message" state with at least two roles:
    persistent continuity storage and transmissible trigger transport.

    v4 separates those roles:
    - fast: exposed local cycle surface
    - slow: deep continuity reservoir
    - carrier: persistent continuity trace
    - packet: transmissible trigger / recruitment surface

    The architecture intentionally keeps both gated and bypass paths explicit so
    "induce", "cheat", and "suppress" can be studied as routing regimes rather
    than treated as errors.
    """

    def __init__(
        self,
        hidden_size: int,
        slow_size: Optional[int] = None,
        carrier_dim: Optional[int] = None,
        packet_dim: Optional[int] = None,
        slow_gate_bias: float = -1.5,
        carrier_gate_bias: float = -0.9,
        packet_gate_bias: float = -1.1,
        fast_mix: float = 0.42,
        slow_step_scale: float = 0.32,
        carrier_step_scale: float = 0.7,
        packet_step_scale: float = 0.55,
        carrier_self_retention: float = 0.48,
        packet_self_retention: float = 0.18,
        packet_capture_scale: float = 0.26,
        slow_carry_scale: float = 0.05,
        packet_drive_scale: float = 0.12,
        fast_update_bias: float = 0.1,
        coupling_to_fast: float = 0.04,
        coupling_to_packet: float = 1.0,
        carrier_init_scale: Optional[float] = None,
        packet_init_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.slow_size = slow_size or hidden_size
        self.carrier_dim = carrier_dim or max(4, hidden_size // 8)
        self.packet_dim = packet_dim or self.carrier_dim
        self.slow_gate_bias = slow_gate_bias
        self.carrier_gate_bias = carrier_gate_bias
        self.packet_gate_bias = packet_gate_bias
        self.fast_mix = fast_mix
        self.slow_step_scale = slow_step_scale
        self.carrier_step_scale = carrier_step_scale
        self.packet_step_scale = packet_step_scale
        self.carrier_self_retention = carrier_self_retention
        self.packet_self_retention = packet_self_retention
        self.packet_capture_scale = packet_capture_scale
        self.slow_carry_scale = slow_carry_scale
        self.packet_drive_scale = packet_drive_scale
        self.fast_update_bias = fast_update_bias
        self.coupling_to_fast = coupling_to_fast
        self.coupling_to_packet = coupling_to_packet
        self.carrier_init_scale = carrier_init_scale if carrier_init_scale is not None else self.init_scale
        self.packet_init_scale = packet_init_scale if packet_init_scale is not None else self.init_scale

        full_dim = hidden_size + self.slow_size + self.carrier_dim + self.packet_dim
        self.fast_in = nn.Linear(full_dim, hidden_size)
        self.slow_in = nn.Linear(hidden_size + self.carrier_dim + self.packet_dim, self.slow_size)
        self.fast_cell = nn.GRUCell(hidden_size, hidden_size)
        self.slow_cell = nn.GRUCell(self.slow_size, self.slow_size)

        self.slow_write_gate = nn.Linear(full_dim, self.slow_size)
        self.carrier_write_gate = nn.Linear(full_dim, self.carrier_dim)
        self.packet_write_gate = nn.Linear(full_dim, self.packet_dim)
        self.fast_update_gate = nn.Linear(full_dim, hidden_size)

        self.carrier_proj = nn.Linear(hidden_size + self.slow_size + self.packet_dim, self.carrier_dim)
        self.packet_proj = nn.Linear(hidden_size + self.slow_size + self.carrier_dim, self.packet_dim)
        self.packet_drive_proj = nn.Linear(hidden_size + self.slow_size + self.carrier_dim, self.packet_dim)

        self.slow_to_fast = nn.Linear(self.slow_size, hidden_size, bias=False)
        self.carrier_to_fast = nn.Linear(self.carrier_dim, hidden_size, bias=False)
        self.packet_to_fast = nn.Linear(self.packet_dim, hidden_size, bias=False)
        self.fast_to_slow = nn.Linear(hidden_size + self.carrier_dim, self.slow_size, bias=False)
        self.carrier_to_slow = nn.Linear(self.carrier_dim, self.slow_size, bias=False)
        self.packet_to_slow = nn.Linear(self.packet_dim, self.slow_size, bias=False)
        self.packet_to_carrier = nn.Linear(self.packet_dim, self.carrier_dim, bias=False)
        self.carrier_self_proj = nn.Linear(self.carrier_dim, self.carrier_dim, bias=False)
        self.packet_self_proj = nn.Linear(self.packet_dim, self.packet_dim, bias=False)

        self.slow_readout = nn.Linear(self.slow_size, hidden_size, bias=False)
        self.carrier_readout = nn.Linear(self.carrier_dim, hidden_size, bias=False)
        self.packet_readout = nn.Linear(self.packet_dim, hidden_size, bias=False)
        self.expose_gate = nn.Linear(hidden_size * 4, hidden_size)
        self.coupling_proj = nn.Linear(hidden_size, self.packet_dim, bias=False)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.slow_size, device=device) * self.init_scale
        carrier = torch.randn(batch_size, self.carrier_dim, device=device) * self.carrier_init_scale
        packet = torch.randn(batch_size, self.packet_dim, device=device) * self.packet_init_scale
        return fast, slow, carrier, packet

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, carrier, packet = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": carrier,
            "packet": packet,
            # Compatibility alias so existing harnesses can compare the new
            # continuity carrier against older message-channel metrics.
            "message": carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, carrier, packet = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.carrier_readout(carrier))
        packet_view = torch.tanh(self.packet_readout(packet))
        gate = torch.sigmoid(
            self.expose_gate(torch.cat([fast, slow_view, carrier_view, packet_view], dim=-1))
        )
        anchored = 0.45 * slow_view + 0.35 * carrier_view + 0.20 * packet_view
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet = state
        stacked = torch.cat([fast, slow, carrier, packet], dim=-1)

        fast_x = self.feedback_scale * self.fast_in(stacked)
        fast_candidate = self.fast_cell(fast_x, fast)

        slow_write = torch.sigmoid(self.slow_write_gate(stacked) + self.slow_gate_bias)
        slow_write = self.slow_step_scale * slow_write
        slow_x = self.feedback_scale * self.slow_in(torch.cat([fast_candidate, carrier, packet], dim=-1))
        slow_candidate = self.slow_cell(slow_x, slow)
        slow_residual = 0.18 * torch.tanh(self.fast_to_slow(torch.cat([fast_candidate, carrier], dim=-1)))
        slow_carry = self.slow_carry_scale * (
            torch.tanh(self.packet_to_slow(packet)) + 0.5 * torch.tanh(self.carrier_to_slow(carrier))
        )
        new_slow = (1.0 - slow_write) * slow + slow_write * (slow_candidate + slow_residual) + slow_carry

        carrier_write = torch.sigmoid(self.carrier_write_gate(stacked) + self.carrier_gate_bias)
        carrier_write = self.carrier_step_scale * carrier_write
        carrier_candidate = torch.tanh(self.carrier_proj(torch.cat([fast_candidate, new_slow, packet], dim=-1)))
        carrier_gated = (1.0 - carrier_write) * carrier + carrier_write * carrier_candidate
        carrier_residual = self.carrier_self_retention * torch.tanh(self.carrier_self_proj(carrier))
        packet_capture = self.packet_capture_scale * torch.tanh(self.packet_to_carrier(packet))
        new_carrier = carrier_gated + carrier_residual + packet_capture

        packet_write = torch.sigmoid(self.packet_write_gate(stacked) + self.packet_gate_bias)
        packet_write = self.packet_step_scale * packet_write
        packet_candidate = torch.tanh(self.packet_proj(torch.cat([fast_candidate, new_slow, new_carrier], dim=-1)))
        packet_gated = (1.0 - packet_write) * packet + packet_write * packet_candidate
        packet_residual = self.packet_self_retention * torch.tanh(self.packet_self_proj(packet))
        packet_drive = self.packet_drive_scale * torch.tanh(
            self.packet_drive_proj(torch.cat([fast, slow, new_carrier], dim=-1))
        )
        new_packet = packet_gated + packet_residual + packet_drive

        fast_anchor = (
            fast_candidate
            + self.fast_mix * torch.tanh(self.slow_to_fast(new_slow))
            + 0.35 * torch.tanh(self.carrier_to_fast(new_carrier))
            + 0.22 * torch.tanh(self.packet_to_fast(new_packet))
        )
        fast_update = torch.sigmoid(self.fast_update_gate(torch.cat([fast_candidate, new_slow, new_carrier, new_packet], dim=-1)) + self.fast_update_bias)
        fast_target = self.state_gain * torch.tanh(fast_anchor)
        new_fast = (1.0 - fast_update) * fast + fast_update * fast_target

        self._step_aux = {
            "fast_update_mean": float(fast_update.mean().item()),
            "slow_write_mean": float(slow_write.mean().item()),
            # Compatibility alias: the current harness treats "message" as the
            # persistent continuity channel. In v4 that role is "carrier".
            "message_write_mean": float(carrier_write.mean().item()),
            "carrier_write_mean": float(carrier_write.mean().item()),
            "packet_write_mean": float(packet_write.mean().item()),
            "slow_carry_norm": float(torch.norm(slow_carry).item()),
            "carrier_residual_norm": float(torch.norm(carrier_residual).item()),
            "packet_capture_norm": float(torch.norm(packet_capture).item()),
            "packet_residual_norm": float(torch.norm(packet_residual).item()),
            "packet_drive_norm": float(torch.norm(packet_drive).item()),
        }
        return new_fast, new_slow, new_carrier, new_packet

    def write_surface_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet = state
        return surface.view(1, -1), slow, carrier, packet

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet = state
        packet_delta = self.coupling_to_packet * strength * torch.tanh(self.coupling_proj(message.view(1, -1)))
        fast_delta = self.coupling_to_fast * strength * message.view(1, -1)
        return fast + fast_delta, slow, carrier, packet + packet_delta


class DualGRUV4MemorySubstrate(DualGRUV4Substrate):
    """v4 plus a simple persistent memory bank.

    The environment remains static. The only added resource is retained internal
    structure. This lets us ask whether the substrate keeps making the same
    routing choice, or whether richer retained information changes which
    self-maintaining path it inhabits.
    """

    def __init__(
        self,
        hidden_size: int,
        memory_dim: Optional[int] = None,
        memory_step_scale: float = 0.25,
        memory_self_retention: float = 0.82,
        memory_to_carrier_scale: float = 0.22,
        memory_to_packet_scale: float = 0.10,
        memory_to_slow_scale: float = 0.05,
        memory_read_scale: float = 0.25,
        memory_init_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.memory_dim = memory_dim or self.carrier_dim
        self.memory_step_scale = memory_step_scale
        self.memory_self_retention = memory_self_retention
        self.memory_to_carrier_scale = memory_to_carrier_scale
        self.memory_to_packet_scale = memory_to_packet_scale
        self.memory_to_slow_scale = memory_to_slow_scale
        self.memory_read_scale = memory_read_scale
        self.memory_init_scale = memory_init_scale if memory_init_scale is not None else self.init_scale

        full_dim = hidden_size + self.slow_size + self.carrier_dim + self.packet_dim + self.memory_dim
        self.memory_fast_in = nn.Linear(full_dim, hidden_size)
        self.memory_slow_in = nn.Linear(hidden_size + self.carrier_dim + self.packet_dim + self.memory_dim, self.slow_size)
        self.memory_slow_write_gate = nn.Linear(full_dim, self.slow_size)
        self.memory_carrier_write_gate = nn.Linear(full_dim, self.carrier_dim)
        self.memory_packet_write_gate = nn.Linear(full_dim, self.packet_dim)
        self.memory_fast_update_gate = nn.Linear(full_dim, hidden_size)
        self.memory_bank_write_gate = nn.Linear(full_dim, self.memory_dim)

        self.memory_bank_proj = nn.Linear(hidden_size + self.slow_size + self.carrier_dim + self.packet_dim, self.memory_dim)
        self.memory_self_proj = nn.Linear(self.memory_dim, self.memory_dim, bias=False)
        self.memory_to_fast = nn.Linear(self.memory_dim, hidden_size, bias=False)
        self.memory_to_carrier = nn.Linear(self.memory_dim, self.carrier_dim, bias=False)
        self.memory_to_packet = nn.Linear(self.memory_dim, self.packet_dim, bias=False)
        self.memory_to_slow = nn.Linear(self.memory_dim, self.slow_size, bias=False)
        self.memory_readout = nn.Linear(self.memory_dim, hidden_size, bias=False)
        self.memory_expose_gate = nn.Linear(hidden_size * 5, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet = super().initial_state(batch_size, device)
        memory = torch.randn(batch_size, self.memory_dim, device=device) * self.memory_init_scale
        return fast, slow, carrier, packet, memory

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, carrier, packet, memory = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": carrier,
            "packet": packet,
            "memory": memory,
            "message": carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, carrier, packet, memory = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.carrier_readout(carrier))
        packet_view = torch.tanh(self.packet_readout(packet))
        memory_view = self.memory_read_scale * torch.tanh(self.memory_readout(memory))
        gate = torch.sigmoid(
            self.memory_expose_gate(torch.cat([fast, slow_view, carrier_view, packet_view, memory_view], dim=-1))
        )
        anchored = 0.35 * slow_view + 0.30 * carrier_view + 0.15 * packet_view + 0.20 * memory_view
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory = state
        stacked = torch.cat([fast, slow, carrier, packet, memory], dim=-1)

        fast_x = self.feedback_scale * self.memory_fast_in(stacked)
        fast_candidate = self.fast_cell(fast_x, fast)

        slow_write = torch.sigmoid(self.memory_slow_write_gate(stacked) + self.slow_gate_bias)
        slow_write = self.slow_step_scale * slow_write
        slow_x = self.feedback_scale * self.memory_slow_in(torch.cat([fast_candidate, carrier, packet, memory], dim=-1))
        slow_candidate = self.slow_cell(slow_x, slow)
        slow_residual = 0.18 * torch.tanh(self.fast_to_slow(torch.cat([fast_candidate, carrier], dim=-1)))
        slow_carry = self.slow_carry_scale * (
            torch.tanh(self.packet_to_slow(packet))
            + 0.5 * torch.tanh(self.carrier_to_slow(carrier))
            + self.memory_to_slow_scale * torch.tanh(self.memory_to_slow(memory))
        )
        new_slow = (1.0 - slow_write) * slow + slow_write * (slow_candidate + slow_residual) + slow_carry

        carrier_write = torch.sigmoid(self.memory_carrier_write_gate(stacked) + self.carrier_gate_bias)
        carrier_write = self.carrier_step_scale * carrier_write
        carrier_candidate = torch.tanh(self.carrier_proj(torch.cat([fast_candidate, new_slow, packet], dim=-1)))
        carrier_gated = (1.0 - carrier_write) * carrier + carrier_write * carrier_candidate
        carrier_residual = self.carrier_self_retention * torch.tanh(self.carrier_self_proj(carrier))
        packet_capture = self.packet_capture_scale * torch.tanh(self.packet_to_carrier(packet))
        memory_carry = self.memory_to_carrier_scale * torch.tanh(self.memory_to_carrier(memory))
        new_carrier = carrier_gated + carrier_residual + packet_capture + memory_carry

        packet_write = torch.sigmoid(self.memory_packet_write_gate(stacked) + self.packet_gate_bias)
        packet_write = self.packet_step_scale * packet_write
        packet_candidate = torch.tanh(self.packet_proj(torch.cat([fast_candidate, new_slow, new_carrier], dim=-1)))
        packet_gated = (1.0 - packet_write) * packet + packet_write * packet_candidate
        packet_residual = self.packet_self_retention * torch.tanh(self.packet_self_proj(packet))
        packet_drive = self.packet_drive_scale * torch.tanh(
            self.packet_drive_proj(torch.cat([fast, slow, new_carrier], dim=-1))
        )
        memory_packet = self.memory_to_packet_scale * torch.tanh(self.memory_to_packet(memory))
        new_packet = packet_gated + packet_residual + packet_drive + memory_packet

        memory_write = torch.sigmoid(self.memory_bank_write_gate(stacked))
        memory_write = self.memory_step_scale * memory_write
        memory_candidate = torch.tanh(self.memory_bank_proj(torch.cat([fast_candidate, new_slow, new_carrier, new_packet], dim=-1)))
        memory_residual = self.memory_self_retention * torch.tanh(self.memory_self_proj(memory))
        new_memory = (1.0 - memory_write) * memory + memory_write * memory_candidate + memory_residual

        fast_anchor = (
            fast_candidate
            + self.fast_mix * torch.tanh(self.slow_to_fast(new_slow))
            + 0.30 * torch.tanh(self.carrier_to_fast(new_carrier))
            + 0.18 * torch.tanh(self.packet_to_fast(new_packet))
            + 0.18 * torch.tanh(self.memory_to_fast(new_memory))
        )
        fast_update = torch.sigmoid(self.memory_fast_update_gate(torch.cat([fast_candidate, new_slow, new_carrier, new_packet, new_memory], dim=-1)) + self.fast_update_bias)
        fast_target = self.state_gain * torch.tanh(fast_anchor)
        new_fast = (1.0 - fast_update) * fast + fast_update * fast_target

        self._step_aux = {
            "fast_update_mean": float(fast_update.mean().item()),
            "slow_write_mean": float(slow_write.mean().item()),
            "message_write_mean": float(carrier_write.mean().item()),
            "carrier_write_mean": float(carrier_write.mean().item()),
            "packet_write_mean": float(packet_write.mean().item()),
            "memory_write_mean": float(memory_write.mean().item()),
            "slow_carry_norm": float(torch.norm(slow_carry).item()),
            "carrier_residual_norm": float(torch.norm(carrier_residual).item()),
            "packet_capture_norm": float(torch.norm(packet_capture).item()),
            "packet_residual_norm": float(torch.norm(packet_residual).item()),
            "packet_drive_norm": float(torch.norm(packet_drive).item()),
            "memory_residual_norm": float(torch.norm(memory_residual).item()),
            "memory_to_carrier_norm": float(torch.norm(memory_carry).item()),
            "memory_to_packet_norm": float(torch.norm(memory_packet).item()),
        }
        return new_fast, new_slow, new_carrier, new_packet, new_memory

    def write_surface_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory = state
        return surface.view(1, -1), slow, carrier, packet, memory

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory = state
        packet_delta = self.coupling_to_packet * strength * torch.tanh(self.coupling_proj(message.view(1, -1)))
        fast_delta = self.coupling_to_fast * strength * message.view(1, -1)
        return fast + fast_delta, slow, carrier, packet + packet_delta, memory


class DualGRUV5Substrate(DualGRUV4MemorySubstrate):
    """v5: endogenous control layer with dual control memories.

    The substrate owns its own control process:
    - control_short tracks local opening pressure and active route heat
    - control_long stores slower route-history and recruitment preference
    - an internal release gate decides when packet pressure should be emitted

    This is not an external optimizer. It is a recurrent control anatomy whose
    only job is to shape endogenous openings, routes, recruitment, and packet
    release from within the system itself.
    """

    def __init__(
        self,
        hidden_size: int,
        control_dim: Optional[int] = None,
        control_short_step_scale: float = 0.28,
        control_long_step_scale: float = 0.16,
        control_short_retention: float = 0.38,
        control_long_retention: float = 0.88,
        carrier_short_retention_scale: float = 0.22,
        carrier_long_retention_scale: float = 0.34,
        release_threshold: float = 0.55,
        release_gain: float = 0.22,
        control_to_packet_scale: float = 0.34,
        control_to_carrier_scale: float = 0.22,
        control_to_slow_scale: float = 0.05,
        control_read_scale: float = 0.22,
        control_init_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.control_dim = control_dim or self.packet_dim
        self.control_short_step_scale = control_short_step_scale
        self.control_long_step_scale = control_long_step_scale
        self.control_short_retention = control_short_retention
        self.control_long_retention = control_long_retention
        self.carrier_short_retention_scale = carrier_short_retention_scale
        self.carrier_long_retention_scale = carrier_long_retention_scale
        self.release_threshold = release_threshold
        self.release_gain = release_gain
        self.control_to_packet_scale = control_to_packet_scale
        self.control_to_carrier_scale = control_to_carrier_scale
        self.control_to_slow_scale = control_to_slow_scale
        self.control_read_scale = control_read_scale
        self.control_init_scale = control_init_scale if control_init_scale is not None else self.init_scale
        # Current v5 calibration:
        # - reduce short-control rewrite pressure
        # - strengthen control routing into packet/carrier
        # - keep endogenous release moderate rather than maximal

        full_dim = (
            hidden_size
            + self.slow_size
            + self.carrier_dim
            + self.packet_dim
            + self.memory_dim
            + self.control_dim
            + self.control_dim
        )
        self.v5_fast_in = nn.Linear(full_dim, hidden_size)
        self.v5_slow_in = nn.Linear(
            hidden_size + self.carrier_dim + self.packet_dim + self.memory_dim + self.control_dim + self.control_dim,
            self.slow_size,
        )
        self.v5_slow_write_gate = nn.Linear(full_dim, self.slow_size)
        self.v5_carrier_write_gate = nn.Linear(full_dim, self.carrier_dim)
        self.v5_packet_write_gate = nn.Linear(full_dim, self.packet_dim)
        self.v5_memory_write_gate = nn.Linear(full_dim, self.memory_dim)
        self.v5_fast_update_gate = nn.Linear(full_dim, hidden_size)

        self.control_short_write_gate = nn.Linear(full_dim, self.control_dim)
        self.control_long_write_gate = nn.Linear(full_dim, self.control_dim)
        self.control_short_proj = nn.Linear(
            hidden_size + self.slow_size + self.carrier_dim + self.packet_dim + self.memory_dim,
            self.control_dim,
        )
        self.control_long_proj = nn.Linear(
            hidden_size + self.slow_size + self.carrier_dim + self.packet_dim + self.memory_dim + self.control_dim,
            self.control_dim,
        )
        self.control_short_self_proj = nn.Linear(self.control_dim, self.control_dim, bias=False)
        self.control_long_self_proj = nn.Linear(self.control_dim, self.control_dim, bias=False)
        self.carrier_short_self_proj = nn.Linear(self.carrier_dim + self.control_dim, self.carrier_dim, bias=False)
        self.carrier_long_self_proj = nn.Linear(self.carrier_dim + self.control_dim, self.carrier_dim, bias=False)

        self.release_gate = nn.Linear(full_dim, 1)
        self.control_to_packet = nn.Linear(self.control_dim * 2, self.packet_dim, bias=False)
        self.control_to_carrier = nn.Linear(self.control_dim * 2, self.carrier_dim, bias=False)
        self.control_to_slow = nn.Linear(self.control_dim * 2, self.slow_size, bias=False)
        self.control_to_fast = nn.Linear(self.control_dim * 2, hidden_size, bias=False)
        self.control_short_readout = nn.Linear(self.control_dim, hidden_size, bias=False)
        self.control_long_readout = nn.Linear(self.control_dim, hidden_size, bias=False)
        self.control_expose_gate = nn.Linear(hidden_size * 7, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory = super().initial_state(batch_size, device)
        control_short = torch.randn(batch_size, self.control_dim, device=device) * self.control_init_scale
        control_long = torch.randn(batch_size, self.control_dim, device=device) * self.control_init_scale
        return fast, slow, carrier, packet, memory, control_short, control_long

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, carrier, packet, memory, control_short, control_long = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": carrier,
            "packet": packet,
            "memory": memory,
            "control_short": control_short,
            "control_long": control_long,
            "message": carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, carrier, packet, memory, control_short, control_long = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.carrier_readout(carrier))
        packet_view = torch.tanh(self.packet_readout(packet))
        memory_view = self.memory_read_scale * torch.tanh(self.memory_readout(memory))
        cs_view = self.control_read_scale * torch.tanh(self.control_short_readout(control_short))
        cl_view = self.control_read_scale * torch.tanh(self.control_long_readout(control_long))
        gate = torch.sigmoid(
            self.control_expose_gate(torch.cat([fast, slow_view, carrier_view, packet_view, memory_view, cs_view, cl_view], dim=-1))
        )
        anchored = 0.28 * slow_view + 0.24 * carrier_view + 0.10 * packet_view + 0.18 * memory_view + 0.12 * cs_view + 0.08 * cl_view
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory, control_short, control_long = state
        stacked = torch.cat([fast, slow, carrier, packet, memory, control_short, control_long], dim=-1)

        fast_x = self.feedback_scale * self.v5_fast_in(stacked)
        fast_candidate = self.fast_cell(fast_x, fast)

        slow_write = torch.sigmoid(self.v5_slow_write_gate(stacked) + self.slow_gate_bias)
        slow_write = self.slow_step_scale * slow_write
        slow_x = self.feedback_scale * self.v5_slow_in(
            torch.cat([fast_candidate, carrier, packet, memory, control_short, control_long], dim=-1)
        )
        slow_candidate = self.slow_cell(slow_x, slow)

        control_pair = torch.cat([control_short, control_long], dim=-1)
        control_slow = self.control_to_slow_scale * torch.tanh(self.control_to_slow(control_pair))
        slow_residual = 0.18 * torch.tanh(self.fast_to_slow(torch.cat([fast_candidate, carrier], dim=-1)))
        slow_carry = self.slow_carry_scale * (
            torch.tanh(self.packet_to_slow(packet))
            + 0.5 * torch.tanh(self.carrier_to_slow(carrier))
            + self.memory_to_slow_scale * torch.tanh(self.memory_to_slow(memory))
        )
        new_slow = (1.0 - slow_write) * slow + slow_write * (slow_candidate + slow_residual) + slow_carry + control_slow

        carrier_write = torch.sigmoid(self.v5_carrier_write_gate(stacked) + self.carrier_gate_bias)
        carrier_write = self.carrier_step_scale * carrier_write
        carrier_candidate = torch.tanh(self.carrier_proj(torch.cat([fast_candidate, new_slow, packet], dim=-1)))
        carrier_gated = (1.0 - carrier_write) * carrier + carrier_write * carrier_candidate
        carrier_short_residual = self.carrier_short_retention_scale * torch.tanh(
            self.carrier_short_self_proj(torch.cat([carrier, control_short], dim=-1))
        )
        carrier_long_residual = self.carrier_long_retention_scale * torch.tanh(
            self.carrier_long_self_proj(torch.cat([carrier, control_long], dim=-1))
        )
        packet_capture = self.packet_capture_scale * torch.tanh(self.packet_to_carrier(packet))
        memory_carry = self.memory_to_carrier_scale * torch.tanh(self.memory_to_carrier(memory))
        control_carry = self.control_to_carrier_scale * torch.tanh(self.control_to_carrier(control_pair))
        new_carrier = carrier_gated + carrier_short_residual + carrier_long_residual + packet_capture + memory_carry + control_carry

        packet_write = torch.sigmoid(self.v5_packet_write_gate(stacked) + self.packet_gate_bias)
        packet_write = self.packet_step_scale * packet_write
        packet_candidate = torch.tanh(self.packet_proj(torch.cat([fast_candidate, new_slow, new_carrier], dim=-1)))
        packet_gated = (1.0 - packet_write) * packet + packet_write * packet_candidate
        packet_residual = self.packet_self_retention * torch.tanh(self.packet_self_proj(packet))
        packet_drive = self.packet_drive_scale * torch.tanh(
            self.packet_drive_proj(torch.cat([fast, slow, new_carrier], dim=-1))
        )
        memory_packet = self.memory_to_packet_scale * torch.tanh(self.memory_to_packet(memory))
        control_packet = self.control_to_packet_scale * torch.tanh(self.control_to_packet(control_pair))
        pre_release_packet = packet_gated + packet_residual + packet_drive + memory_packet + control_packet

        release_logit = self.release_gate(stacked)
        release_open = torch.sigmoid(release_logit)
        release_strength = self.release_gain * torch.relu(release_open - self.release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(self.state_vector(state)))
        new_packet = pre_release_packet + endogenous_release

        memory_write = torch.sigmoid(self.v5_memory_write_gate(stacked))
        memory_write = self.memory_step_scale * memory_write
        memory_candidate = torch.tanh(self.memory_bank_proj(torch.cat([fast_candidate, new_slow, new_carrier, new_packet], dim=-1)))
        memory_residual = self.memory_self_retention * torch.tanh(self.memory_self_proj(memory))
        new_memory = (1.0 - memory_write) * memory + memory_write * memory_candidate + memory_residual

        # Endogenous control memories: short tracks present opening pressure,
        # long tracks route history and whether release was productive.
        route_pressure = torch.cat([fast_candidate, new_slow, new_carrier, new_packet, new_memory], dim=-1)
        control_short_write = torch.sigmoid(self.control_short_write_gate(stacked))
        control_short_write = self.control_short_step_scale * control_short_write
        control_short_candidate = torch.tanh(self.control_short_proj(route_pressure))
        control_short_residual = self.control_short_retention * torch.tanh(self.control_short_self_proj(control_short))
        new_control_short = (1.0 - control_short_write) * control_short + control_short_write * control_short_candidate + control_short_residual

        long_pressure = torch.cat([route_pressure, new_control_short], dim=-1)
        control_long_write = torch.sigmoid(self.control_long_write_gate(stacked))
        control_long_write = self.control_long_step_scale * control_long_write
        control_long_candidate = torch.tanh(self.control_long_proj(long_pressure))
        control_long_residual = self.control_long_retention * torch.tanh(self.control_long_self_proj(control_long))
        new_control_long = (1.0 - control_long_write) * control_long + control_long_write * control_long_candidate + control_long_residual

        fast_anchor = (
            fast_candidate
            + self.fast_mix * torch.tanh(self.slow_to_fast(new_slow))
            + 0.30 * torch.tanh(self.carrier_to_fast(new_carrier))
            + 0.16 * torch.tanh(self.packet_to_fast(new_packet))
            + 0.16 * torch.tanh(self.memory_to_fast(new_memory))
            + 0.14 * torch.tanh(self.control_to_fast(torch.cat([new_control_short, new_control_long], dim=-1)))
        )
        fast_update = torch.sigmoid(self.v5_fast_update_gate(
            torch.cat([fast_candidate, new_slow, new_carrier, new_packet, new_memory, new_control_short, new_control_long], dim=-1)
        ) + self.fast_update_bias)
        fast_target = self.state_gain * torch.tanh(fast_anchor)
        new_fast = (1.0 - fast_update) * fast + fast_update * fast_target

        self._step_aux = {
            "fast_update_mean": float(fast_update.mean().item()),
            "slow_write_mean": float(slow_write.mean().item()),
            "message_write_mean": float(carrier_write.mean().item()),
            "carrier_write_mean": float(carrier_write.mean().item()),
            "packet_write_mean": float(packet_write.mean().item()),
            "memory_write_mean": float(memory_write.mean().item()),
            "control_short_write_mean": float(control_short_write.mean().item()),
            "control_long_write_mean": float(control_long_write.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "slow_carry_norm": float(torch.norm(slow_carry).item()),
            "carrier_residual_norm": float(torch.norm(carrier_short_residual + carrier_long_residual).item()),
            "carrier_short_residual_norm": float(torch.norm(carrier_short_residual).item()),
            "carrier_long_residual_norm": float(torch.norm(carrier_long_residual).item()),
            "packet_capture_norm": float(torch.norm(packet_capture).item()),
            "packet_residual_norm": float(torch.norm(packet_residual).item()),
            "packet_drive_norm": float(torch.norm(packet_drive).item()),
            "memory_residual_norm": float(torch.norm(memory_residual).item()),
            "memory_to_carrier_norm": float(torch.norm(memory_carry).item()),
            "memory_to_packet_norm": float(torch.norm(memory_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_carry).item()),
            "control_to_packet_norm": float(torch.norm(control_packet).item()),
            "control_to_slow_norm": float(torch.norm(control_slow).item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
        }
        return new_fast, new_slow, new_carrier, new_packet, new_memory, new_control_short, new_control_long

    def write_surface_state(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory, control_short, control_long = state
        return surface.view(1, -1), slow, carrier, packet, memory, control_short, control_long

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, carrier, packet, memory, control_short, control_long = state
        packet_delta = self.coupling_to_packet * strength * torch.tanh(self.coupling_proj(message.view(1, -1)))
        fast_delta = self.coupling_to_fast * strength * message.view(1, -1)
        return fast + fast_delta, slow, carrier, packet + packet_delta, memory, control_short, control_long


class DemianNativeV0Substrate(SelfLoopSubstrate):
    """First custom substrate without GRU cells.

    Channels:
    - fast: exposed local surface
    - slow: deep continuity basin
    - long_carrier: long-lived persistence owner
    - short_support: short-lived route stabilizer
    - packet: transmissible recruitment surface
    - control_short: local onset / burst pressure
    - control_long: slower route-history memory

    Design goal:
    keep only the structural distinctions that survived the GRU-family probes,
    and express them as explicit route-based updates rather than inherited cell
    mechanics.
    """

    def __init__(
        self,
        hidden_size: int,
        carrier_dim: Optional[int] = None,
        support_dim: Optional[int] = None,
        packet_dim: Optional[int] = None,
        control_dim: Optional[int] = None,
        slow_decay: float = 0.92,
        long_carrier_decay: float = 0.95,
        short_support_decay: float = 0.72,
        packet_decay: float = 0.78,
        control_short_decay: float = 0.68,
        control_long_decay: float = 0.94,
        release_threshold: float = 0.56,
        release_gain: float = 0.18,
        control_to_packet_scale: float = 0.30,
        control_to_carrier_scale: float = 0.24,
        control_to_slow_scale: float = 0.08,
        support_to_packet_scale: float = 0.16,
        packet_to_carrier_scale: float = 0.18,
        packet_to_slow_scale: float = 0.08,
        slow_to_fast_scale: float = 0.28,
        carrier_to_fast_scale: float = 0.24,
        support_to_fast_scale: float = 0.14,
        control_to_fast_scale: float = 0.12,
        control_read_scale: float = 0.18,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.carrier_dim = carrier_dim or max(4, hidden_size // 8)
        self.support_dim = support_dim or self.carrier_dim
        self.packet_dim = packet_dim or self.carrier_dim
        self.control_dim = control_dim or self.packet_dim
        self.slow_decay = slow_decay
        self.long_carrier_decay = long_carrier_decay
        self.short_support_decay = short_support_decay
        self.packet_decay = packet_decay
        self.control_short_decay = control_short_decay
        self.control_long_decay = control_long_decay
        self.release_threshold = release_threshold
        self.release_gain = release_gain
        self.control_to_packet_scale = control_to_packet_scale
        self.control_to_carrier_scale = control_to_carrier_scale
        self.control_to_slow_scale = control_to_slow_scale
        self.support_to_packet_scale = support_to_packet_scale
        self.packet_to_carrier_scale = packet_to_carrier_scale
        self.packet_to_slow_scale = packet_to_slow_scale
        self.slow_to_fast_scale = slow_to_fast_scale
        self.carrier_to_fast_scale = carrier_to_fast_scale
        self.support_to_fast_scale = support_to_fast_scale
        self.control_to_fast_scale = control_to_fast_scale
        self.control_read_scale = control_read_scale

        full_dim = (
            hidden_size
            + hidden_size
            + self.carrier_dim
            + self.support_dim
            + self.packet_dim
            + self.control_dim
            + self.control_dim
        )
        self.fast_mix = nn.Linear(full_dim, hidden_size)
        self.slow_mix = nn.Linear(full_dim, hidden_size)
        self.long_carrier_mix = nn.Linear(full_dim, self.carrier_dim)
        self.short_support_mix = nn.Linear(full_dim, self.support_dim)
        self.packet_mix = nn.Linear(full_dim, self.packet_dim)
        self.control_short_mix = nn.Linear(full_dim, self.control_dim)
        self.control_long_mix = nn.Linear(full_dim, self.control_dim)

        self.fast_gate = nn.Linear(full_dim, hidden_size)
        self.slow_gate = nn.Linear(full_dim, hidden_size)
        self.long_carrier_gate = nn.Linear(full_dim, self.carrier_dim)
        self.short_support_gate = nn.Linear(full_dim, self.support_dim)
        self.packet_gate = nn.Linear(full_dim, self.packet_dim)
        self.control_short_gate = nn.Linear(full_dim, self.control_dim)
        self.control_long_gate = nn.Linear(full_dim, self.control_dim)
        self.release_gate = nn.Linear(full_dim, 1)

        self.slow_readout = nn.Linear(hidden_size, hidden_size, bias=False)
        self.long_carrier_readout = nn.Linear(self.carrier_dim, hidden_size, bias=False)
        self.short_support_readout = nn.Linear(self.support_dim, hidden_size, bias=False)
        self.packet_readout = nn.Linear(self.packet_dim, hidden_size, bias=False)
        self.control_short_readout = nn.Linear(self.control_dim, hidden_size, bias=False)
        self.control_long_readout = nn.Linear(self.control_dim, hidden_size, bias=False)
        self.expose_gate = nn.Linear(hidden_size * 6, hidden_size)
        self.coupling_proj = nn.Linear(hidden_size, self.packet_dim, bias=False)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        long_carrier = torch.randn(batch_size, self.carrier_dim, device=device) * self.init_scale
        short_support = torch.randn(batch_size, self.support_dim, device=device) * self.init_scale
        packet = torch.randn(batch_size, self.packet_dim, device=device) * self.init_scale
        control_short = torch.randn(batch_size, self.control_dim, device=device) * self.init_scale
        control_long = torch.randn(batch_size, self.control_dim, device=device) * self.init_scale
        return fast, slow, long_carrier, short_support, packet, control_short, control_long

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": long_carrier,
            "short_support": short_support,
            "packet": packet,
            "control_short": control_short,
            "control_long": control_long,
            "message": long_carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.long_carrier_readout(long_carrier))
        support_view = torch.tanh(self.short_support_readout(short_support))
        packet_view = torch.tanh(self.packet_readout(packet))
        cs_view = self.control_read_scale * torch.tanh(self.control_short_readout(control_short))
        cl_view = self.control_read_scale * torch.tanh(self.control_long_readout(control_long))
        gate = torch.sigmoid(self.expose_gate(torch.cat([fast, slow_view, carrier_view, support_view, packet_view, cs_view + cl_view], dim=-1)))
        anchored = 0.24 * slow_view + 0.28 * carrier_view + 0.14 * support_view + 0.12 * packet_view + 0.12 * cs_view + 0.10 * cl_view
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        control_pair = torch.cat([control_short, control_long], dim=-1)
        control_to_packet = self.control_to_packet_scale * torch.tanh(control_pair[:, : self.packet_dim] + control_pair[:, self.control_dim : self.control_dim + self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_pair[:, : self.carrier_dim] + control_pair[:, self.control_dim : self.control_dim + self.carrier_dim])
        control_to_slow = self.control_to_slow_scale * torch.tanh(slow_candidate)
        support_to_packet = self.support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        packet_to_slow = self.packet_to_slow_scale * torch.tanh(self.slow_readout(slow) + self.long_carrier_readout(long_carrier))

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = self.release_gain * torch.relu(release_open - self.release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate

        new_short_support = self.short_support_decay * short_support + support_gate * support_candidate + 0.12 * torch.tanh(control_short[:, : self.support_dim])
        new_packet = self.packet_decay * packet + packet_gate * packet_candidate + support_to_packet + control_to_packet + endogenous_release
        new_long_carrier = self.long_carrier_decay * long_carrier + carrier_gate * carrier_candidate + packet_to_carrier + control_to_carrier + 0.08 * torch.tanh(new_control_long[:, : self.carrier_dim])
        new_slow = self.slow_decay * slow + slow_gate * slow_candidate + packet_to_slow + control_to_slow + 0.10 * torch.tanh(self.slow_readout(slow))

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long

    def write_surface_state(
        self,
        state: tuple[torch.Tensor, ...],
        surface: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        _fast, slow, long_carrier, short_support, packet, control_short, control_long, *tail = state
        return surface.view(1, -1), slow, long_carrier, short_support, packet, control_short, control_long, *tail

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, ...],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, ...]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, *tail = state
        packet_delta = strength * torch.tanh(self.coupling_proj(message.view(1, -1)))
        return fast, slow, long_carrier, short_support, packet + packet_delta, control_short, control_long, *tail


class DemianNativeV1Substrate(DemianNativeV0Substrate):
    """Native substrate refinement with stronger slow-control ownership.

    Keeps the v0 channel anatomy but simplifies the transition logic around
    the routes that most consistently predicted earlier recruitment:
    direct control into slow, long-carrier persistence, and moderated packet
    dependence.
    """

    def __init__(
        self,
        hidden_size: int,
        control_to_packet_scale: float = 0.24,
        control_to_carrier_scale: float = 0.30,
        control_to_slow_scale: float = 0.16,
        support_to_packet_scale: float = 0.06,
        packet_to_carrier_scale: float = 0.12,
        packet_to_slow_scale: float = 0.12,
        long_carrier_decay: float = 0.97,
        short_support_decay: float = 0.64,
        control_long_decay: float = 0.96,
        release_threshold: float = 0.58,
        release_gain: float = 0.16,
        carrier_to_slow_scale: float = 0.16,
        control_long_to_carrier_scale: float = 0.12,
        support_feedback_scale: float = 0.04,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            control_to_packet_scale=control_to_packet_scale,
            control_to_carrier_scale=control_to_carrier_scale,
            control_to_slow_scale=control_to_slow_scale,
            support_to_packet_scale=support_to_packet_scale,
            packet_to_carrier_scale=packet_to_carrier_scale,
            packet_to_slow_scale=packet_to_slow_scale,
            long_carrier_decay=long_carrier_decay,
            short_support_decay=short_support_decay,
            control_long_decay=control_long_decay,
            release_threshold=release_threshold,
            release_gain=release_gain,
            **kwargs,
        )
        self.carrier_to_slow_scale = carrier_to_slow_scale
        self.control_long_to_carrier_scale = control_long_to_carrier_scale
        self.support_feedback_scale = support_feedback_scale

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        control_bridge = torch.tanh(control_short + 0.65 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))

        control_to_packet = self.control_to_packet_scale * torch.tanh(
            control_bridge[:, : self.packet_dim]
        )
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(
            control_bridge[:, : self.carrier_dim]
        )
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(
            long_control_bridge[:, : self.carrier_dim]
        )
        carrier_to_slow = self.carrier_to_slow_scale * carrier_readout
        control_to_slow = self.control_to_slow_scale * torch.tanh(
            slow_candidate + carrier_readout + self.control_long_readout(control_long)
        )
        support_to_packet = self.support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        packet_to_slow = self.packet_to_slow_scale * torch.tanh(
            self.packet_readout(packet) + carrier_readout
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = self.release_gain * torch.relu(release_open - self.release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate

        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            self.long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.06 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long


class DemianNativeV2Substrate(DemianNativeV1Substrate):
    """Native refinement with explicit recruitment bias into slow.

    Relative to v1:
    - less generic release openness / support traffic
    - stronger packet+carrier bridging into slow
    - slower long-carrier decay so recruitment has a more stable owner
    """

    def __init__(
        self,
        hidden_size: int,
        control_to_packet_scale: float = 0.20,
        control_to_carrier_scale: float = 0.32,
        control_to_slow_scale: float = 0.18,
        support_to_packet_scale: float = 0.03,
        packet_to_carrier_scale: float = 0.10,
        packet_to_slow_scale: float = 0.18,
        long_carrier_decay: float = 0.98,
        short_support_decay: float = 0.58,
        release_threshold: float = 0.62,
        release_gain: float = 0.12,
        carrier_to_slow_scale: float = 0.22,
        control_long_to_carrier_scale: float = 0.14,
        support_feedback_scale: float = 0.02,
        **kwargs,
    ):
        super().__init__(
            hidden_size,
            control_to_packet_scale=control_to_packet_scale,
            control_to_carrier_scale=control_to_carrier_scale,
            control_to_slow_scale=control_to_slow_scale,
            support_to_packet_scale=support_to_packet_scale,
            packet_to_carrier_scale=packet_to_carrier_scale,
            packet_to_slow_scale=packet_to_slow_scale,
            long_carrier_decay=long_carrier_decay,
            short_support_decay=short_support_decay,
            release_threshold=release_threshold,
            release_gain=release_gain,
            carrier_to_slow_scale=carrier_to_slow_scale,
            control_long_to_carrier_scale=control_long_to_carrier_scale,
            support_feedback_scale=support_feedback_scale,
            **kwargs,
        )

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))

        control_to_packet = self.control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = self.support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = self.carrier_to_slow_scale * carrier_readout
        packet_to_slow = self.packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = self.control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = self.release_gain * torch.relu(release_open - self.release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            self.long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(self.long_carrier_decay * long_carrier).item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long


class DemianNativeV3Substrate(DemianNativeV2Substrate):
    """Native v2 plus endogenous tightness control.

    The substrate should not choose between "v1-like sensitivity" and
    "v2-like consolidation" globally. Instead it maintains a small internal
    tightness state that modulates how strongly retained structure gets
    recruited into slow ownership versus allowed to circulate more loosely.

    Low tightness:
    - preserve weak perturbation sensitivity
    - allow more support / packet circulation
    - keep carrier recruitment into slow weaker

    High tightness:
    - consolidate carrier / packet structure into slow more strongly
    - reduce generic release and support traffic
    - slow carrier loss
    """

    def __init__(
        self,
        hidden_size: int,
        tightness_retention: float = 0.90,
        initial_tightness_bias: float = -0.35,
        tightness_to_carrier_slow_scale: float = 0.22,
        tightness_to_packet_slow_scale: float = 0.18,
        tightness_to_control_slow_scale: float = 0.12,
        tightness_to_release_gain_scale: float = 0.08,
        tightness_to_release_threshold_scale: float = 0.08,
        tightness_to_support_packet_scale: float = 0.025,
        tightness_to_carrier_decay_scale: float = 0.018,
        tightness_to_control_packet_scale: float = 0.035,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.tightness_retention = tightness_retention
        self.initial_tightness_bias = initial_tightness_bias
        self.tightness_to_carrier_slow_scale = tightness_to_carrier_slow_scale
        self.tightness_to_packet_slow_scale = tightness_to_packet_slow_scale
        self.tightness_to_control_slow_scale = tightness_to_control_slow_scale
        self.tightness_to_release_gain_scale = tightness_to_release_gain_scale
        self.tightness_to_release_threshold_scale = tightness_to_release_threshold_scale
        self.tightness_to_support_packet_scale = tightness_to_support_packet_scale
        self.tightness_to_carrier_decay_scale = tightness_to_carrier_decay_scale
        self.tightness_to_control_packet_scale = tightness_to_control_packet_scale

        full_dim = (
            hidden_size
            + hidden_size
            + self.carrier_dim
            + self.support_dim
            + self.packet_dim
            + self.control_dim
            + self.control_dim
            + 1
        )
        self.tightness_mix = nn.Linear(full_dim, 1)
        self.tightness_readout = nn.Linear(1, hidden_size, bias=False)
        self.tightness_expose_gate = nn.Linear(hidden_size * 7, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = super().initial_state(batch_size, device)
        tightness = torch.full((batch_size, 1), self.initial_tightness_bias, device=device)
        return fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": long_carrier,
            "short_support": short_support,
            "packet": packet,
            "control_short": control_short,
            "control_long": control_long,
            "tightness": tightness,
            "message": long_carrier,
        }

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.long_carrier_readout(long_carrier))
        support_view = torch.tanh(self.short_support_readout(short_support))
        packet_view = torch.tanh(self.packet_readout(packet))
        cs_view = self.control_read_scale * torch.tanh(self.control_short_readout(control_short))
        cl_view = self.control_read_scale * torch.tanh(self.control_long_readout(control_long))
        tight_view = torch.tanh(self.tightness_readout(torch.tanh(tightness)))
        gate = torch.sigmoid(
            self.tightness_expose_gate(
                torch.cat([fast, slow_view, carrier_view, support_view, packet_view, cs_view + cl_view, tight_view], dim=-1)
            )
        )
        anchored = (
            0.22 * slow_view
            + 0.25 * carrier_view
            + 0.12 * support_view
            + 0.10 * packet_view
            + 0.10 * cs_view
            + 0.09 * cl_view
            + 0.12 * tight_view
        )
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        eff_control_to_slow_scale = self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open
        eff_packet_to_slow_scale = self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open
        eff_carrier_to_slow_scale = self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV4Substrate(DemianNativeV3Substrate):
    """Native v3 plus bounded endogenous weight dynamics on the governor.

    The goal is not generic plasticity. The goal is to let the substrate learn
    better governance of its own consolidation versus looseness boundary.

    Plasticity lives only on a small governor path that perturbs tightness
    control. This keeps weight learning:
    - local
    - slow
    - bounded
    - inspectable
    """

    def __init__(
        self,
        hidden_size: int,
        plasticity_eta: float = 8e-5,
        plasticity_decay: float = 8e-4,
        plasticity_oja: float = 0.08,
        plasticity_max_norm: float = 0.25,
        plasticity_drive_scale: float = 0.08,
        plasticity_gate_threshold: float = 0.42,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        full_dim = (
            hidden_size
            + hidden_size
            + self.carrier_dim
            + self.support_dim
            + self.packet_dim
            + self.control_dim
            + self.control_dim
            + 1
        )
        self.plasticity_eta = plasticity_eta
        self.plasticity_decay = plasticity_decay
        self.plasticity_oja = plasticity_oja
        self.plasticity_max_norm = plasticity_max_norm
        self.plasticity_drive_scale = plasticity_drive_scale
        self.plasticity_gate_threshold = plasticity_gate_threshold
        self.plastic_tightness_mix = nn.Linear(full_dim, 1, bias=False)
        with torch.no_grad():
            self.plastic_tightness_mix.weight.zero_()

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        governor_in = torch.cat([stacked, tightness], dim=-1)
        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        plasticity_evidence = (
            0.42 * carrier_pressure
            + 0.24 * packet_pressure
            + 0.18 * control_pressure
            + 0.16 * coherence_signal
        )
        plasticity_gate = torch.sigmoid(8.0 * (plasticity_evidence - self.plasticity_gate_threshold))
        normalized_governor = governor_in / math.sqrt(governor_in.shape[-1])
        plastic_drive_raw = self.plastic_tightness_mix(normalized_governor)
        plastic_drive = self.plasticity_drive_scale * plasticity_gate * torch.tanh(plastic_drive_raw)
        tightness_drive = torch.tanh(self.tightness_mix(governor_in) + plastic_drive)
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        eff_control_to_slow_scale = self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open
        eff_packet_to_slow_scale = self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open
        eff_carrier_to_slow_scale = self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        # Local Oja-style plasticity on the governor path.
        with torch.no_grad():
            pre = normalized_governor.detach()
            pre = pre / pre.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            consolidation_target = 2.0 * plasticity_gate.detach() - 1.0
            post = consolidation_target
            post_mean = post.mean(dim=0, keepdim=True)
            gate_mean = float(plasticity_gate.detach().mean().item())
            hebb = post.T @ pre
            oja_term = self.plasticity_oja * float(post_mean.pow(2).mean().item()) * self.plastic_tightness_mix.weight.data
            self.plastic_tightness_mix.weight.data.mul_(1.0 - self.plasticity_decay * gate_mean)
            self.plastic_tightness_mix.weight.data.add_(self.plasticity_eta * hebb)
            self.plastic_tightness_mix.weight.data.sub_(self.plasticity_eta * oja_term)
            w_norm = float(torch.norm(self.plastic_tightness_mix.weight.data).item())
            if w_norm > self.plasticity_max_norm:
                self.plastic_tightness_mix.weight.data.mul_(self.plasticity_max_norm / (w_norm + 1e-10))

        plastic_weight_norm = float(torch.norm(self.plastic_tightness_mix.weight).item())
        plastic_drive_mean = float(plastic_drive.mean().item())

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "plastic_weight_norm": plastic_weight_norm,
            "plastic_drive_mean": plastic_drive_mean,
            "plasticity_gate_mean": float(plasticity_gate.mean().item()),
            "plasticity_evidence_mean": float(plasticity_evidence.mean().item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV5Substrate(DemianNativeV3Substrate):
    """Native v3 plus route-local plasticity on slow-governing paths."""

    def __init__(
        self,
        hidden_size: int,
        route_plasticity_eta: float = 1.5e-3,
        route_trace_decay: float = 0.92,
        route_weight_decay: float = 0.015,
        route_delta_max: float = 0.10,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.route_plasticity_eta = route_plasticity_eta
        self.route_trace_decay = route_trace_decay
        self.route_weight_decay = route_weight_decay
        self.route_delta_max = route_delta_max
        self.register_buffer("route_eligibility", torch.zeros(3))
        self.register_buffer("route_scale_delta", torch.zeros(3))

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.route_eligibility.zero_()
            self.route_scale_delta.zero_()
        return super().initial_state(batch_size, device)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        learned_control_delta = float(self.route_scale_delta[0].item())
        learned_packet_delta = float(self.route_scale_delta[1].item())
        learned_carrier_delta = float(self.route_scale_delta[2].item())
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale
            + self.tightness_to_control_slow_scale * tightness_open
            + learned_control_delta,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale
            + self.tightness_to_packet_slow_scale * tightness_open
            + learned_packet_delta,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale
            + self.tightness_to_carrier_slow_scale * tightness_open
            + learned_carrier_delta,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=self.route_eligibility.device,
            )
            self.route_eligibility.mul_(self.route_trace_decay).add_((1.0 - self.route_trace_decay) * route_activity)
            consolidation_signal = (
                0.36 * float(carrier_pressure.mean().item())
                + 0.22 * float(packet_pressure.mean().item())
                + 0.22 * float(coherence_signal.mean().item())
                + 0.20 * float(slow_gate.mean().item())
            )
            destabilization_signal = (
                0.34 * float(state_mismatch.mean().item())
                + 0.28 * float(release_strength.mean().item())
                + 0.20 * float(control_pressure.mean().item())
                + 0.18 * float(loose_open.mean().item())
            )
            modulation = math.tanh(2.5 * (consolidation_signal - destabilization_signal))
            self.route_scale_delta.mul_(1.0 - self.route_weight_decay)
            self.route_scale_delta.add_(self.route_plasticity_eta * modulation * self.route_eligibility)
            self.route_scale_delta.clamp_(-self.route_delta_max, self.route_delta_max)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "route_plastic_modulation_mean": modulation,
            "route_control_eligibility_mean": float(self.route_eligibility[0].item()),
            "route_packet_eligibility_mean": float(self.route_eligibility[1].item()),
            "route_carrier_eligibility_mean": float(self.route_eligibility[2].item()),
            "route_control_delta_mean": float(self.route_scale_delta[0].item()),
            "route_packet_delta_mean": float(self.route_scale_delta[1].item()),
            "route_carrier_delta_mean": float(self.route_scale_delta[2].item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV51Substrate(DemianNativeV3Substrate):
    """Native v5.1: route-local plasticity with learned endogenous credit."""

    def __init__(
        self,
        hidden_size: int,
        route_plasticity_eta: float = 1.5e-3,
        route_trace_decay: float = 0.92,
        route_weight_decay: float = 0.015,
        route_delta_max: float = 0.10,
        credit_eta: float = 2.5e-3,
        credit_decay: float = 0.01,
        credit_max_norm: float = 0.6,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.route_plasticity_eta = route_plasticity_eta
        self.route_trace_decay = route_trace_decay
        self.route_weight_decay = route_weight_decay
        self.route_delta_max = route_delta_max
        self.credit_eta = credit_eta
        self.credit_decay = credit_decay
        self.credit_max_norm = credit_max_norm
        self.register_buffer("route_eligibility", torch.zeros(3))
        self.register_buffer("route_scale_delta", torch.zeros(3))
        self.register_buffer("credit_prev_coherence", torch.zeros(1))
        self.register_buffer("credit_prev_mismatch", torch.zeros(1))
        self.register_buffer("credit_prev_release", torch.zeros(1))
        self.register_buffer("credit_prev_route_mean", torch.zeros(1))
        self.register_buffer("credit_bootstrap", torch.zeros(1))
        self.register_buffer("credit_initialized", torch.zeros(1))
        self.credit_head = nn.Linear(11, 1)
        with torch.no_grad():
            self.credit_head.weight.zero_()
            self.credit_head.bias.zero_()

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.route_eligibility.zero_()
            self.route_scale_delta.zero_()
            self.credit_prev_coherence.zero_()
            self.credit_prev_mismatch.zero_()
            self.credit_prev_release.zero_()
            self.credit_prev_route_mean.zero_()
            self.credit_bootstrap.zero_()
            self.credit_initialized.zero_()
        return super().initial_state(batch_size, device)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        batch_size = stacked.shape[0]

        credit_features = torch.cat(
            [
                self.route_eligibility.unsqueeze(0).expand(batch_size, -1),
                self.route_scale_delta.unsqueeze(0).expand(batch_size, -1),
                carrier_pressure,
                packet_pressure,
                control_pressure,
                coherence_signal,
                loose_open,
            ],
            dim=-1,
        )
        credit_pred = torch.tanh(self.credit_head(credit_features))
        modulation = float(credit_pred.mean().item())

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        learned_control_delta = float(self.route_scale_delta[0].item())
        learned_packet_delta = float(self.route_scale_delta[1].item())
        learned_carrier_delta = float(self.route_scale_delta[2].item())
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open + learned_control_delta,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open + learned_packet_delta,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open + learned_carrier_delta,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=self.route_eligibility.device,
            )
            self.route_eligibility.mul_(self.route_trace_decay).add_((1.0 - self.route_trace_decay) * route_activity)

            coherence_now = float(coherence_signal.mean().item())
            mismatch_now = float(state_mismatch.mean().item())
            release_now = float(release_strength.mean().item())
            route_mean_now = float(route_activity.mean().item())
            route_balance = 1.0 - float(route_activity.std().item()) / (route_mean_now + 1e-6)
            route_balance = max(-1.0, min(1.0, route_balance))

            if float(self.credit_initialized.item()) > 0.5:
                viability_delta = (
                    0.34 * (coherence_now - float(self.credit_prev_coherence.item()))
                    + 0.26 * (float(self.credit_prev_mismatch.item()) - mismatch_now)
                    + 0.18 * (float(self.credit_prev_release.item()) - release_now)
                    + 0.12 * (route_mean_now - float(self.credit_prev_route_mean.item()))
                    + 0.10 * route_balance
                )
            else:
                viability_delta = 0.18 * coherence_now + 0.12 * route_balance - 0.10 * mismatch_now
                self.credit_initialized.fill_(1.0)

            target = math.tanh(3.0 * viability_delta + 0.35 * float(self.credit_bootstrap.item()))
            pred_mean = float(credit_pred.mean().item())
            error = target - pred_mean
            feat_mean = credit_features.detach().mean(dim=0, keepdim=True)
            self.credit_head.weight.data.mul_(1.0 - self.credit_decay)
            self.credit_head.bias.data.mul_(1.0 - self.credit_decay)
            self.credit_head.weight.data.add_(self.credit_eta * error * feat_mean)
            self.credit_head.bias.data.add_(self.credit_eta * error)
            w_norm = float(torch.norm(self.credit_head.weight.data).item())
            if w_norm > self.credit_max_norm:
                self.credit_head.weight.data.mul_(self.credit_max_norm / (w_norm + 1e-10))

            self.route_scale_delta.mul_(1.0 - self.route_weight_decay)
            self.route_scale_delta.add_(self.route_plasticity_eta * pred_mean * self.route_eligibility)
            self.route_scale_delta.clamp_(-self.route_delta_max, self.route_delta_max)

            self.credit_prev_coherence.fill_(coherence_now)
            self.credit_prev_mismatch.fill_(mismatch_now)
            self.credit_prev_release.fill_(release_now)
            self.credit_prev_route_mean.fill_(route_mean_now)
            self.credit_bootstrap.fill_(pred_mean)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "route_plastic_modulation_mean": pred_mean,
            "route_control_eligibility_mean": float(self.route_eligibility[0].item()),
            "route_packet_eligibility_mean": float(self.route_eligibility[1].item()),
            "route_carrier_eligibility_mean": float(self.route_eligibility[2].item()),
            "route_control_delta_mean": float(self.route_scale_delta[0].item()),
            "route_packet_delta_mean": float(self.route_scale_delta[1].item()),
            "route_carrier_delta_mean": float(self.route_scale_delta[2].item()),
            "credit_prediction_mean": pred_mean,
            "credit_target_mean": target,
            "credit_error_mean": error,
            "credit_weight_norm": float(torch.norm(self.credit_head.weight).item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV52Substrate(DemianNativeV3Substrate):
    """Native v5.2: delayed self-derived credit with inherited weak priors."""

    def __init__(
        self,
        hidden_size: int,
        route_plasticity_eta: float = 1.2e-3,
        route_trace_decay: float = 0.94,
        route_weight_decay: float = 0.012,
        route_delta_max: float = 0.10,
        credit_eta: float = 1.0e-3,
        credit_decay: float = 0.004,
        credit_max_norm: float = 0.6,
        credit_update_interval: int = 8,
        credit_delay: int = 6,
        inherited_credit_bias: tuple[float, float, float, float, float, float, float, float, float, float, float] = (
            0.10, 0.08, 0.12, 0.06, 0.05, 0.08, 0.07, 0.05, 0.09, -0.06, -0.05
        ),
        inherited_credit_scale: float = 0.08,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.route_plasticity_eta = route_plasticity_eta
        self.route_trace_decay = route_trace_decay
        self.route_weight_decay = route_weight_decay
        self.route_delta_max = route_delta_max
        self.credit_eta = credit_eta
        self.credit_decay = credit_decay
        self.credit_max_norm = credit_max_norm
        self.credit_update_interval = credit_update_interval
        self.credit_delay = credit_delay
        self.inherited_credit_scale = inherited_credit_scale
        self.register_buffer("route_eligibility", torch.zeros(3))
        self.register_buffer("route_scale_delta", torch.zeros(3))
        self.register_buffer("credit_step", torch.zeros(1))
        self.register_buffer("credit_bootstrap", torch.zeros(1))
        self.register_buffer("credit_initialized", torch.zeros(1))
        self.credit_head = nn.Linear(11, 1)
        with torch.no_grad():
            prior = torch.tensor(inherited_credit_bias, dtype=self.credit_head.weight.dtype)
            self.credit_head.weight.copy_(self.inherited_credit_scale * prior.unsqueeze(0))
            self.credit_head.bias.zero_()
        self._credit_history: list[dict[str, float]] = []

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.route_eligibility.zero_()
            self.route_scale_delta.zero_()
            self.credit_step.zero_()
            self.credit_bootstrap.zero_()
            self.credit_initialized.zero_()
        self._credit_history = []
        return super().initial_state(batch_size, device)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        batch_size = stacked.shape[0]

        credit_features = torch.cat(
            [
                self.route_eligibility.unsqueeze(0).expand(batch_size, -1),
                self.route_scale_delta.unsqueeze(0).expand(batch_size, -1),
                carrier_pressure,
                packet_pressure,
                control_pressure,
                coherence_signal,
                loose_open,
            ],
            dim=-1,
        )
        credit_pred = torch.tanh(self.credit_head(credit_features))
        pred_mean = float(credit_pred.mean().item())

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        learned_control_delta = float(self.route_scale_delta[0].item())
        learned_packet_delta = float(self.route_scale_delta[1].item())
        learned_carrier_delta = float(self.route_scale_delta[2].item())
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open + learned_control_delta,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open + learned_packet_delta,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open + learned_carrier_delta,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=self.route_eligibility.device,
            )
            self.route_eligibility.mul_(self.route_trace_decay).add_((1.0 - self.route_trace_decay) * route_activity)

            coherence_now = float(coherence_signal.mean().item())
            mismatch_now = float(state_mismatch.mean().item())
            release_now = float(release_strength.mean().item())
            route_mean_now = float(route_activity.mean().item())
            route_balance = 1.0 - float(route_activity.std().item()) / (route_mean_now + 1e-6)
            route_balance = max(-1.0, min(1.0, route_balance))

            history_row = {
                "coherence": coherence_now,
                "mismatch": mismatch_now,
                "release": release_now,
                "route_mean": route_mean_now,
                "route_balance": route_balance,
                "pred": pred_mean,
                "features": credit_features.detach().mean(dim=0, keepdim=True),
            }
            self._credit_history.append(history_row)
            max_keep = max(self.credit_delay + self.credit_update_interval + 4, 32)
            if len(self._credit_history) > max_keep:
                self._credit_history = self._credit_history[-max_keep:]

            self.credit_step.add_(1.0)
            target = pred_mean
            error = 0.0
            if (
                len(self._credit_history) > self.credit_delay
                and int(self.credit_step.item()) % self.credit_update_interval == 0
            ):
                past = self._credit_history[-(self.credit_delay + 1)]
                viability_delta = (
                    0.34 * (coherence_now - past["coherence"])
                    + 0.24 * (past["mismatch"] - mismatch_now)
                    + 0.16 * (past["release"] - release_now)
                    + 0.14 * (route_mean_now - past["route_mean"])
                    + 0.12 * route_balance
                )
                target = math.tanh(2.5 * viability_delta + 0.30 * float(self.credit_bootstrap.item()))
                error = target - pred_mean
                feat_mean = past["features"]
                self.credit_head.weight.data.mul_(1.0 - self.credit_decay)
                self.credit_head.bias.data.mul_(1.0 - self.credit_decay)
                self.credit_head.weight.data.add_(self.credit_eta * error * feat_mean)
                self.credit_head.bias.data.add_(self.credit_eta * error)
                w_norm = float(torch.norm(self.credit_head.weight.data).item())
                if w_norm > self.credit_max_norm:
                    self.credit_head.weight.data.mul_(self.credit_max_norm / (w_norm + 1e-10))
                self.credit_bootstrap.fill_(pred_mean)
                self.credit_initialized.fill_(1.0)

            self.route_scale_delta.mul_(1.0 - self.route_weight_decay)
            self.route_scale_delta.add_(self.route_plasticity_eta * pred_mean * self.route_eligibility)
            self.route_scale_delta.clamp_(-self.route_delta_max, self.route_delta_max)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "route_plastic_modulation_mean": pred_mean,
            "route_control_eligibility_mean": float(self.route_eligibility[0].item()),
            "route_packet_eligibility_mean": float(self.route_eligibility[1].item()),
            "route_carrier_eligibility_mean": float(self.route_eligibility[2].item()),
            "route_control_delta_mean": float(self.route_scale_delta[0].item()),
            "route_packet_delta_mean": float(self.route_scale_delta[1].item()),
            "route_carrier_delta_mean": float(self.route_scale_delta[2].item()),
            "credit_prediction_mean": pred_mean,
            "credit_target_mean": float(target),
            "credit_error_mean": float(error),
            "credit_weight_norm": float(torch.norm(self.credit_head.weight).item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV52BSubstrate(DemianNativeV3Substrate):
    """Native v5.2b: delayed credit with explicit anti-locking controls."""

    def __init__(
        self,
        hidden_size: int,
        route_plasticity_eta: float = 1.0e-3,
        route_trace_decay: float = 0.94,
        route_weight_decay: float = 0.014,
        route_delta_max: float = 0.10,
        credit_eta: float = 1.0e-3,
        credit_decay: float = 0.005,
        credit_max_norm: float = 0.6,
        credit_update_interval: int = 8,
        credit_delay: int = 6,
        challenge_interval: int = 128,
        challenge_length: int = 12,
        challenge_start_step: int = 0,
        lock_challenge_threshold: float = 0.0,
        challenge_decay_boost: float = 0.10,
        challenge_lr_scale: float = 0.25,
        inherited_credit_bias: tuple[float, float, float, float, float, float, float, float, float, float, float] = (
            0.10, 0.08, 0.12, 0.06, 0.05, 0.08, 0.07, 0.05, 0.09, -0.06, -0.05
        ),
        inherited_credit_scale: float = 0.08,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.route_plasticity_eta = route_plasticity_eta
        self.route_trace_decay = route_trace_decay
        self.route_weight_decay = route_weight_decay
        self.route_delta_max = route_delta_max
        self.credit_eta = credit_eta
        self.credit_decay = credit_decay
        self.credit_max_norm = credit_max_norm
        self.credit_update_interval = credit_update_interval
        self.credit_delay = credit_delay
        self.challenge_interval = challenge_interval
        self.challenge_length = challenge_length
        self.challenge_start_step = challenge_start_step
        self.lock_challenge_threshold = lock_challenge_threshold
        self.challenge_decay_boost = challenge_decay_boost
        self.challenge_lr_scale = challenge_lr_scale
        self.inherited_credit_scale = inherited_credit_scale
        self.register_buffer("route_eligibility", torch.zeros(3))
        self.register_buffer("route_scale_delta", torch.zeros(3))
        self.register_buffer("credit_step", torch.zeros(1))
        self.register_buffer("credit_bootstrap", torch.zeros(1))
        self.register_buffer("credit_initialized", torch.zeros(1))
        self.register_buffer("pred_ema", torch.zeros(1))
        self.register_buffer("pred_change_ema", torch.zeros(1))
        self.register_buffer("lock_risk_ema", torch.zeros(1))
        self.credit_head = nn.Linear(11, 1)
        with torch.no_grad():
            prior = torch.tensor(inherited_credit_bias, dtype=self.credit_head.weight.dtype)
            self.credit_head.weight.copy_(self.inherited_credit_scale * prior.unsqueeze(0))
            self.credit_head.bias.zero_()
        self._credit_history: list[dict[str, float]] = []

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.route_eligibility.zero_()
            self.route_scale_delta.zero_()
            self.credit_step.zero_()
            self.credit_bootstrap.zero_()
            self.credit_initialized.zero_()
            self.pred_ema.zero_()
            self.pred_change_ema.zero_()
            self.lock_risk_ema.zero_()
        self._credit_history = []
        return super().initial_state(batch_size, device)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        batch_size = stacked.shape[0]

        credit_features = torch.cat(
            [
                self.route_eligibility.unsqueeze(0).expand(batch_size, -1),
                self.route_scale_delta.unsqueeze(0).expand(batch_size, -1),
                carrier_pressure,
                packet_pressure,
                control_pressure,
                coherence_signal,
                loose_open,
            ],
            dim=-1,
        )
        credit_pred = torch.tanh(self.credit_head(credit_features))
        pred_mean = float(credit_pred.mean().item())

        step_idx = int(self.credit_step.item())
        challenge_phase = step_idx > 0 and (step_idx % self.challenge_interval) < self.challenge_length
        route_norm = float(torch.norm(self.route_scale_delta).item()) / (self.route_delta_max * math.sqrt(3.0) + 1e-6)
        pred_change = abs(pred_mean - float(self.pred_ema.item()))
        with torch.no_grad():
            self.pred_change_ema.mul_(0.96).add_(0.04 * pred_change)
            self.pred_ema.mul_(0.94).add_(0.06 * pred_mean)
        low_change = max(0.0, 0.03 - float(self.pred_change_ema.item())) / 0.03
        lock_risk = 1.0 / (
            1.0
            + math.exp(
                -6.0
                * (
                    0.50 * route_norm
                    + 0.30 * max(0.0, pred_mean)
                    + 0.20 * low_change
                    - 0.45
                )
            )
        )
        if challenge_phase:
            lock_risk = min(1.0, lock_risk + 0.15)
        with torch.no_grad():
            self.lock_risk_ema.mul_(0.95).add_(0.05 * lock_risk)
        in_challenge = (
            challenge_phase
            and step_idx >= self.challenge_start_step
            and float(self.lock_risk_ema.item()) >= self.lock_challenge_threshold
        )

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        learned_control_delta = float(self.route_scale_delta[0].item())
        learned_packet_delta = float(self.route_scale_delta[1].item())
        learned_carrier_delta = float(self.route_scale_delta[2].item())
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open + learned_control_delta,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open + learned_packet_delta,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open + learned_carrier_delta,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=self.route_eligibility.device,
            )
            self.route_eligibility.mul_(self.route_trace_decay).add_((1.0 - self.route_trace_decay) * route_activity)

            coherence_now = float(coherence_signal.mean().item())
            mismatch_now = float(state_mismatch.mean().item())
            release_now = float(release_strength.mean().item())
            route_mean_now = float(route_activity.mean().item())
            route_balance = 1.0 - float(route_activity.std().item()) / (route_mean_now + 1e-6)
            route_balance = max(-1.0, min(1.0, route_balance))

            history_row = {
                "coherence": coherence_now,
                "mismatch": mismatch_now,
                "release": release_now,
                "route_mean": route_mean_now,
                "route_balance": route_balance,
                "pred": pred_mean,
                "features": credit_features.detach().mean(dim=0, keepdim=True),
            }
            self._credit_history.append(history_row)
            max_keep = max(self.credit_delay + self.credit_update_interval + 4, 32)
            if len(self._credit_history) > max_keep:
                self._credit_history = self._credit_history[-max_keep:]

            self.credit_step.add_(1.0)
            target = pred_mean
            error = 0.0
            effective_credit_decay = self.credit_decay + 0.01 * lock_risk + (0.02 if in_challenge else 0.0)
            if (
                len(self._credit_history) > self.credit_delay
                and int(self.credit_step.item()) % self.credit_update_interval == 0
            ):
                past = self._credit_history[-(self.credit_delay + 1)]
                viability_delta = (
                    0.34 * (coherence_now - past["coherence"])
                    + 0.24 * (past["mismatch"] - mismatch_now)
                    + 0.16 * (past["release"] - release_now)
                    + 0.14 * (route_mean_now - past["route_mean"])
                    + 0.12 * route_balance
                    - 0.18 * lock_risk
                )
                target = math.tanh(2.3 * viability_delta + 0.25 * float(self.credit_bootstrap.item()))
                error = target - pred_mean
                feat_mean = past["features"]
                credit_lr = self.credit_eta * (1.0 - 0.65 * lock_risk) * (0.45 if in_challenge else 1.0)
                self.credit_head.weight.data.mul_(1.0 - effective_credit_decay)
                self.credit_head.bias.data.mul_(1.0 - effective_credit_decay)
                self.credit_head.weight.data.add_(credit_lr * error * feat_mean)
                self.credit_head.bias.data.add_(credit_lr * error)
                w_norm = float(torch.norm(self.credit_head.weight.data).item())
                if w_norm > self.credit_max_norm:
                    self.credit_head.weight.data.mul_(self.credit_max_norm / (w_norm + 1e-10))
                self.credit_bootstrap.fill_(pred_mean)
                self.credit_initialized.fill_(1.0)

            effective_route_decay = self.route_weight_decay + 0.05 * lock_risk + (self.challenge_decay_boost if in_challenge else 0.0)
            effective_route_lr = self.route_plasticity_eta * (1.0 - 0.85 * lock_risk) * (self.challenge_lr_scale if in_challenge else 1.0)
            self.route_scale_delta.mul_(1.0 - effective_route_decay)
            self.route_scale_delta.add_(effective_route_lr * pred_mean * self.route_eligibility)
            self.route_scale_delta.clamp_(-self.route_delta_max, self.route_delta_max)
            if in_challenge:
                self.route_scale_delta.mul_(0.92)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "route_plastic_modulation_mean": pred_mean,
            "route_control_eligibility_mean": float(self.route_eligibility[0].item()),
            "route_packet_eligibility_mean": float(self.route_eligibility[1].item()),
            "route_carrier_eligibility_mean": float(self.route_eligibility[2].item()),
            "route_control_delta_mean": float(self.route_scale_delta[0].item()),
            "route_packet_delta_mean": float(self.route_scale_delta[1].item()),
            "route_carrier_delta_mean": float(self.route_scale_delta[2].item()),
            "credit_prediction_mean": pred_mean,
            "credit_target_mean": float(target),
            "credit_error_mean": float(error),
            "credit_weight_norm": float(torch.norm(self.credit_head.weight).item()),
            "lock_risk_mean": float(lock_risk),
            "challenge_active_mean": float(1.0 if in_challenge else 0.0),
            "effective_route_decay_mean": float(effective_route_decay),
            "effective_route_lr_mean": float(effective_route_lr),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV52CSubstrate(DemianNativeV52BSubstrate):
    """Native v5.2c: later, lock-conditional anti-locking."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            challenge_interval=192,
            challenge_length=10,
            challenge_start_step=384,
            lock_challenge_threshold=0.26,
            challenge_decay_boost=0.06,
            challenge_lr_scale=0.45,
            **kwargs,
        )


class DemianNativeV53Substrate(DemianNativeV3Substrate):
    """Native v5.3: phase-aware credit and anti-locking controller."""

    PHASE_EMERGENCE = 0
    PHASE_CONSOLIDATION = 1
    PHASE_LOCK_RISK = 2
    PHASE_CHALLENGE = 3
    PHASE_RECOVERY = 4

    def __init__(
        self,
        hidden_size: int,
        route_plasticity_eta: float = 1.0e-3,
        route_trace_decay: float = 0.94,
        route_weight_decay: float = 0.014,
        route_delta_max: float = 0.10,
        credit_eta: float = 1.0e-3,
        credit_decay: float = 0.005,
        credit_max_norm: float = 0.6,
        credit_update_interval: int = 8,
        credit_delay: int = 6,
        challenge_interval: int = 192,
        challenge_length: int = 10,
        challenge_start_step: int = 384,
        lock_challenge_threshold: float = 0.26,
        lock_phase_threshold: float = 0.28,
        challenge_decay_boost: float = 0.06,
        challenge_lr_scale: float = 0.45,
        emergence_until: int = 160,
        recovery_length: int = 56,
        inherited_credit_bias: tuple[float, float, float, float, float, float, float, float, float, float, float] = (
            0.10, 0.08, 0.12, 0.06, 0.05, 0.08, 0.07, 0.05, 0.09, -0.06, -0.05
        ),
        inherited_credit_scale: float = 0.08,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.route_plasticity_eta = route_plasticity_eta
        self.route_trace_decay = route_trace_decay
        self.route_weight_decay = route_weight_decay
        self.route_delta_max = route_delta_max
        self.credit_eta = credit_eta
        self.credit_decay = credit_decay
        self.credit_max_norm = credit_max_norm
        self.credit_update_interval = credit_update_interval
        self.credit_delay = credit_delay
        self.challenge_interval = challenge_interval
        self.challenge_length = challenge_length
        self.challenge_start_step = challenge_start_step
        self.lock_challenge_threshold = lock_challenge_threshold
        self.lock_phase_threshold = lock_phase_threshold
        self.challenge_decay_boost = challenge_decay_boost
        self.challenge_lr_scale = challenge_lr_scale
        self.emergence_until = emergence_until
        self.recovery_length = recovery_length
        self.inherited_credit_scale = inherited_credit_scale
        self.register_buffer("route_eligibility", torch.zeros(3))
        self.register_buffer("route_scale_delta", torch.zeros(3))
        self.register_buffer("credit_step", torch.zeros(1))
        self.register_buffer("credit_bootstrap", torch.zeros(1))
        self.register_buffer("pred_ema", torch.zeros(1))
        self.register_buffer("pred_change_ema", torch.zeros(1))
        self.register_buffer("lock_risk_ema", torch.zeros(1))
        self.register_buffer("phase_id", torch.zeros(1))
        self.register_buffer("last_challenge_step", torch.full((1,), -1000.0))
        self.credit_head = nn.Linear(11, 1)
        with torch.no_grad():
            prior = torch.tensor(inherited_credit_bias, dtype=self.credit_head.weight.dtype)
            self.credit_head.weight.copy_(self.inherited_credit_scale * prior.unsqueeze(0))
            self.credit_head.bias.zero_()
        self._credit_history: list[dict[str, float]] = []

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.route_eligibility.zero_()
            self.route_scale_delta.zero_()
            self.credit_step.zero_()
            self.credit_bootstrap.zero_()
            self.pred_ema.zero_()
            self.pred_change_ema.zero_()
            self.lock_risk_ema.zero_()
            self.phase_id.zero_()
            self.last_challenge_step.fill_(-1000.0)
        self._credit_history = []
        return super().initial_state(batch_size, device)

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        batch_size = stacked.shape[0]

        credit_features = torch.cat(
            [
                self.route_eligibility.unsqueeze(0).expand(batch_size, -1),
                self.route_scale_delta.unsqueeze(0).expand(batch_size, -1),
                carrier_pressure,
                packet_pressure,
                control_pressure,
                coherence_signal,
                loose_open,
            ],
            dim=-1,
        )
        credit_pred = torch.tanh(self.credit_head(credit_features))
        pred_mean = float(credit_pred.mean().item())

        step_idx = int(self.credit_step.item())
        route_norm = float(torch.norm(self.route_scale_delta).item()) / (self.route_delta_max * math.sqrt(3.0) + 1e-6)
        pred_change = abs(pred_mean - float(self.pred_ema.item()))
        with torch.no_grad():
            self.pred_change_ema.mul_(0.96).add_(0.04 * pred_change)
            self.pred_ema.mul_(0.94).add_(0.06 * pred_mean)
        low_change = max(0.0, 0.03 - float(self.pred_change_ema.item())) / 0.03
        lock_risk = 1.0 / (
            1.0
            + math.exp(
                -6.0
                * (
                    0.50 * route_norm
                    + 0.30 * max(0.0, pred_mean)
                    + 0.20 * low_change
                    - 0.45
                )
            )
        )
        with torch.no_grad():
            self.lock_risk_ema.mul_(0.95).add_(0.05 * lock_risk)
        challenge_phase = step_idx > 0 and (step_idx % self.challenge_interval) < self.challenge_length
        in_recovery = (step_idx - int(self.last_challenge_step.item())) <= self.recovery_length and step_idx > 0
        if challenge_phase and float(self.lock_risk_ema.item()) >= self.lock_challenge_threshold and step_idx >= self.challenge_start_step:
            phase = self.PHASE_CHALLENGE
            with torch.no_grad():
                self.last_challenge_step.fill_(float(step_idx))
        elif in_recovery:
            phase = self.PHASE_RECOVERY
        elif step_idx < self.emergence_until and route_norm < 0.18 and float(coherence_signal.mean().item()) < 0.70:
            phase = self.PHASE_EMERGENCE
        elif float(self.lock_risk_ema.item()) >= self.lock_phase_threshold or (route_norm >= 0.24 and low_change >= 0.45):
            phase = self.PHASE_LOCK_RISK
        else:
            phase = self.PHASE_CONSOLIDATION
        with torch.no_grad():
            self.phase_id.fill_(float(phase))

        phase_credit_lr_scale = {
            self.PHASE_EMERGENCE: 0.55,
            self.PHASE_CONSOLIDATION: 1.00,
            self.PHASE_LOCK_RISK: 0.60,
            self.PHASE_CHALLENGE: 0.28,
            self.PHASE_RECOVERY: 0.72,
        }[phase]
        phase_route_lr_scale = {
            self.PHASE_EMERGENCE: 0.70,
            self.PHASE_CONSOLIDATION: 1.00,
            self.PHASE_LOCK_RISK: 0.48,
            self.PHASE_CHALLENGE: 0.18,
            self.PHASE_RECOVERY: 0.62,
        }[phase]
        phase_decay_boost = {
            self.PHASE_EMERGENCE: 0.00,
            self.PHASE_CONSOLIDATION: 0.00,
            self.PHASE_LOCK_RISK: 0.05,
            self.PHASE_CHALLENGE: 0.11,
            self.PHASE_RECOVERY: 0.03,
        }[phase]

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        learned_control_delta = float(self.route_scale_delta[0].item())
        learned_packet_delta = float(self.route_scale_delta[1].item())
        learned_carrier_delta = float(self.route_scale_delta[2].item())
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open + learned_control_delta,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open + learned_packet_delta,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open + learned_carrier_delta,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=self.route_eligibility.device,
            )
            self.route_eligibility.mul_(self.route_trace_decay).add_((1.0 - self.route_trace_decay) * route_activity)

            coherence_now = float(coherence_signal.mean().item())
            mismatch_now = float(state_mismatch.mean().item())
            release_now = float(release_strength.mean().item())
            route_mean_now = float(route_activity.mean().item())
            route_balance = 1.0 - float(route_activity.std().item()) / (route_mean_now + 1e-6)
            route_balance = max(-1.0, min(1.0, route_balance))

            history_row = {
                "coherence": coherence_now,
                "mismatch": mismatch_now,
                "release": release_now,
                "route_mean": route_mean_now,
                "route_balance": route_balance,
                "pred": pred_mean,
                "features": credit_features.detach().mean(dim=0, keepdim=True),
            }
            self._credit_history.append(history_row)
            max_keep = max(self.credit_delay + self.credit_update_interval + 4, 32)
            if len(self._credit_history) > max_keep:
                self._credit_history = self._credit_history[-max_keep:]

            self.credit_step.add_(1.0)
            target = pred_mean
            error = 0.0
            effective_credit_decay = self.credit_decay + 0.008 * lock_risk + phase_decay_boost * 0.2
            if (
                len(self._credit_history) > self.credit_delay
                and int(self.credit_step.item()) % self.credit_update_interval == 0
            ):
                past = self._credit_history[-(self.credit_delay + 1)]
                viability_delta = (
                    0.34 * (coherence_now - past["coherence"])
                    + 0.24 * (past["mismatch"] - mismatch_now)
                    + 0.16 * (past["release"] - release_now)
                    + 0.14 * (route_mean_now - past["route_mean"])
                    + 0.12 * route_balance
                    - 0.18 * lock_risk
                )
                target = math.tanh(2.3 * viability_delta + 0.25 * float(self.credit_bootstrap.item()))
                error = target - pred_mean
                feat_mean = past["features"]
                credit_lr = self.credit_eta * (1.0 - 0.55 * lock_risk) * phase_credit_lr_scale
                self.credit_head.weight.data.mul_(1.0 - effective_credit_decay)
                self.credit_head.bias.data.mul_(1.0 - effective_credit_decay)
                self.credit_head.weight.data.add_(credit_lr * error * feat_mean)
                self.credit_head.bias.data.add_(credit_lr * error)
                w_norm = float(torch.norm(self.credit_head.weight.data).item())
                if w_norm > self.credit_max_norm:
                    self.credit_head.weight.data.mul_(self.credit_max_norm / (w_norm + 1e-10))
                self.credit_bootstrap.fill_(pred_mean)

            effective_route_decay = self.route_weight_decay + 0.04 * lock_risk + phase_decay_boost
            effective_route_lr = self.route_plasticity_eta * (1.0 - 0.75 * lock_risk) * phase_route_lr_scale
            self.route_scale_delta.mul_(1.0 - effective_route_decay)
            self.route_scale_delta.add_(effective_route_lr * pred_mean * self.route_eligibility)
            self.route_scale_delta.clamp_(-self.route_delta_max, self.route_delta_max)
            if phase == self.PHASE_CHALLENGE:
                self.route_scale_delta.mul_(0.90)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "route_plastic_modulation_mean": pred_mean,
            "route_control_eligibility_mean": float(self.route_eligibility[0].item()),
            "route_packet_eligibility_mean": float(self.route_eligibility[1].item()),
            "route_carrier_eligibility_mean": float(self.route_eligibility[2].item()),
            "route_control_delta_mean": float(self.route_scale_delta[0].item()),
            "route_packet_delta_mean": float(self.route_scale_delta[1].item()),
            "route_carrier_delta_mean": float(self.route_scale_delta[2].item()),
            "credit_prediction_mean": pred_mean,
            "credit_target_mean": float(target),
            "credit_error_mean": float(error),
            "credit_weight_norm": float(torch.norm(self.credit_head.weight).item()),
            "lock_risk_mean": float(lock_risk),
            "challenge_active_mean": float(1.0 if phase == self.PHASE_CHALLENGE else 0.0),
            "effective_route_decay_mean": float(effective_route_decay),
            "effective_route_lr_mean": float(effective_route_lr),
            "phase_id_mean": float(phase),
            "phase_emergence_mean": float(1.0 if phase == self.PHASE_EMERGENCE else 0.0),
            "phase_consolidation_mean": float(1.0 if phase == self.PHASE_CONSOLIDATION else 0.0),
            "phase_lock_risk_mean": float(1.0 if phase == self.PHASE_LOCK_RISK else 0.0),
            "phase_recovery_mean": float(1.0 if phase == self.PHASE_RECOVERY else 0.0),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV6Substrate(DemianNativeV3Substrate):
    """Native v6: endogenous observer, value, and actuator controller on top of v3."""

    def __init__(
        self,
        hidden_size: int,
        controller_dim: Optional[int] = None,
        conflict_lanes: int = 6,
        topology_edges: int = 6,
        controller_retention: float = 0.92,
        conflict_retention: float = 0.95,
        topology_retention: float = 0.985,
        conflict_drive_scale: float = 0.40,
        conflict_containment_scale: float = 0.16,
        conflict_transform_scale: float = 0.06,
        topology_update_scale: float = 0.035,
        topology_delta_scale: float = 0.08,
        topology_open_only: bool = False,
        topology_carrier_floor: float = 0.0,
        conflict_resolution_scale: Optional[float] = None,
        controller_eta: float = 8.0e-4,
        value_eta: float = 1.0e-3,
        controller_decay: float = 0.004,
        controller_max_norm: float = 0.8,
        actuator_delta_scale: float = 0.10,
        controller_update_interval: int = 8,
        controller_delay: int = 6,
        controller_warmup_steps: int = 64,
        controller_freeze_after_steps: Optional[int] = None,
        actuator_penalty_scale: float = 0.18,
        actuator_commit_limiter: float = 0.75,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.controller_dim = controller_dim or max(4, hidden_size // 8)
        self.conflict_lanes = max(1, conflict_lanes)
        self.topology_edges = max(1, topology_edges)
        self.controller_retention = controller_retention
        self.conflict_retention = conflict_retention
        self.topology_retention = topology_retention
        self.conflict_drive_scale = conflict_drive_scale
        if conflict_resolution_scale is not None:
            conflict_containment_scale = conflict_resolution_scale
        self.conflict_containment_scale = conflict_containment_scale
        self.conflict_transform_scale = conflict_transform_scale
        self.topology_update_scale = topology_update_scale
        self.topology_delta_scale = topology_delta_scale
        self.topology_open_only = topology_open_only
        self.topology_carrier_floor = topology_carrier_floor
        self.controller_eta = controller_eta
        self.value_eta = value_eta
        self.controller_decay = controller_decay
        self.controller_max_norm = controller_max_norm
        self.actuator_delta_scale = actuator_delta_scale
        self.controller_update_interval = controller_update_interval
        self.controller_delay = controller_delay
        self.controller_warmup_steps = controller_warmup_steps
        self.controller_freeze_after_steps = controller_freeze_after_steps
        self.actuator_penalty_scale = actuator_penalty_scale
        self.actuator_commit_limiter = actuator_commit_limiter
        self.controller_horizon_delays = (
            max(1, controller_delay),
            max(2, controller_delay * 4),
            max(4, controller_delay * 12),
        )
        self.controller_horizon_weights = (0.20, 0.30, 0.50)
        self.controller_long_priority_gain = 0.65
        self.controller_actuator_guard_scale = 0.45

        observer_dim = 12 + self.conflict_lanes
        self.controller_observer = nn.Linear(observer_dim, self.controller_dim)
        self.controller_state_mix = nn.Linear(self.controller_dim * 2, self.controller_dim)
        self.controller_state_gate = nn.Linear(self.controller_dim * 2, self.controller_dim)
        self.controller_value_head = nn.Linear(self.controller_dim, 3)
        self.controller_actuator_head = nn.Linear(self.controller_dim, 7)
        self.controller_readout = nn.Linear(self.controller_dim, hidden_size, bias=False)
        self.conflict_readout = nn.Linear(self.conflict_lanes, hidden_size, bias=False)
        self.conflict_to_packet = nn.Linear(self.conflict_lanes, self.packet_dim, bias=False)
        self.conflict_to_carrier = nn.Linear(self.conflict_lanes, self.carrier_dim, bias=False)
        self.conflict_to_slow = nn.Linear(self.conflict_lanes, hidden_size, bias=False)
        self.conflict_to_topology = nn.Linear(self.conflict_lanes, self.topology_edges, bias=False)
        self.topology_readout = nn.Linear(self.topology_edges, hidden_size, bias=False)
        self.v6_expose_gate = nn.Linear(hidden_size * 10, hidden_size)

        self.register_buffer("controller_step", torch.zeros(1))
        self.register_buffer("controller_bootstrap", torch.zeros(3))
        self._controller_history: list[dict[str, object]] = []

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = super().initial_state(
            batch_size, device
        )
        controller_state = torch.zeros(batch_size, self.controller_dim, device=device)
        conflict_potential = torch.zeros(batch_size, self.conflict_lanes, device=device)
        topology_state = torch.zeros(batch_size, self.topology_edges, device=device)
        with torch.no_grad():
            self.controller_step.zero_()
            self.controller_bootstrap.zero_()
        self._controller_history = []
        return fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness, controller_state, conflict_potential, topology_state

    def state_components(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness, controller_state, conflict_potential, topology_state = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": long_carrier,
            "short_support": short_support,
            "packet": packet,
            "control_short": control_short,
            "control_long": control_long,
            "tightness": tightness,
            "controller": controller_state,
            "conflict_potential": conflict_potential,
            "topology": topology_state,
            "message": long_carrier,
        }

    def state_vector(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness, controller_state, conflict_potential, topology_state = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.long_carrier_readout(long_carrier))
        support_view = torch.tanh(self.short_support_readout(short_support))
        packet_view = torch.tanh(self.packet_readout(packet))
        cs_view = self.control_read_scale * torch.tanh(self.control_short_readout(control_short))
        cl_view = self.control_read_scale * torch.tanh(self.control_long_readout(control_long))
        tight_view = torch.tanh(self.tightness_readout(torch.tanh(tightness)))
        controller_view = 0.14 * torch.tanh(self.controller_readout(controller_state))
        conflict_view = 0.10 * torch.tanh(self.conflict_readout(torch.tanh(conflict_potential)))
        topology_view = 0.08 * torch.tanh(self.topology_readout(torch.tanh(topology_state)))
        gate = torch.sigmoid(
            self.v6_expose_gate(
                torch.cat(
                    [fast, slow_view, carrier_view, support_view, packet_view, cs_view + cl_view, tight_view, controller_view, conflict_view, topology_view],
                    dim=-1,
                )
            )
        )
        anchored = (
            0.20 * slow_view
            + 0.23 * carrier_view
            + 0.10 * support_view
            + 0.09 * packet_view
            + 0.09 * cs_view
            + 0.08 * cl_view
            + 0.11 * tight_view
            + 0.10 * controller_view
            + 0.07 * conflict_view
            + 0.04 * topology_view
        )
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness, controller_state, conflict_potential, topology_state = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))
        carrier_pressure = carrier_readout.abs().mean(dim=-1, keepdim=True)
        packet_pressure = packet_readout.abs().mean(dim=-1, keepdim=True)
        control_pressure = control_bridge.abs().mean(dim=-1, keepdim=True)
        state_mismatch = torch.abs(torch.tanh(fast) - torch.tanh(slow)).mean(dim=-1, keepdim=True)
        coherence_signal = 1.0 - state_mismatch.clamp(0.0, 1.0)
        conflict_lane_activation = torch.sigmoid(conflict_potential)

        observer_features = torch.cat(
            [
                carrier_pressure,
                packet_pressure,
                control_pressure,
                coherence_signal,
                tightness_open,
                loose_open,
                conflict_lane_activation,
                torch.tanh(topology_state).abs().mean(dim=-1, keepdim=True),
                slow_gate.mean(dim=-1, keepdim=True),
                packet_gate.mean(dim=-1, keepdim=True),
                carrier_gate.mean(dim=-1, keepdim=True),
                control_short_gate.mean(dim=-1, keepdim=True),
                control_long_gate.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        observer_drive = torch.tanh(self.controller_observer(observer_features))
        controller_in = torch.cat([controller_state, observer_drive], dim=-1)
        controller_gate = torch.sigmoid(self.controller_state_gate(controller_in))
        controller_candidate = torch.tanh(self.controller_state_mix(controller_in))
        step_idx = int(self.controller_step.item())
        warmup_progress = min(1.0, max(0.0, step_idx / max(1, self.controller_warmup_steps)))
        freeze_active = self.controller_freeze_after_steps is not None and step_idx >= self.controller_freeze_after_steps
        effective_commit = self.actuator_commit_limiter * (0.25 + 0.75 * warmup_progress)
        new_controller_state = self.controller_retention * controller_state + effective_commit * controller_gate * controller_candidate

        value_pred_tensor = torch.tanh(self.controller_value_head(new_controller_state))
        value_pred_vector = value_pred_tensor.mean(dim=0)
        value_pred_short = float(value_pred_vector[0].item())
        value_pred_medium = float(value_pred_vector[1].item())
        value_pred_long = float(value_pred_vector[2].item())
        retained_conflict = float(conflict_lane_activation.mean().item())
        horizon_conflict = max(0.0, value_pred_medium - value_pred_long)
        long_priority = min(1.0, self.controller_long_priority_gain * horizon_conflict * max(0.20, 1.0 - 0.75 * retained_conflict))
        eff_short_weight = max(0.05, self.controller_horizon_weights[0] - 0.05 * long_priority)
        eff_medium_weight = max(0.05, self.controller_horizon_weights[1] - 0.22 * long_priority)
        eff_long_weight = self.controller_horizon_weights[2] + 0.27 * long_priority
        eff_weight_sum = eff_short_weight + eff_medium_weight + eff_long_weight
        eff_short_weight /= eff_weight_sum
        eff_medium_weight /= eff_weight_sum
        eff_long_weight /= eff_weight_sum
        value_pred = (
            eff_short_weight * value_pred_short
            + eff_medium_weight * value_pred_medium
            + eff_long_weight * value_pred_long
        )
        raw_actuator = torch.tanh(self.controller_actuator_head(new_controller_state))
        actuator_scale = self.actuator_delta_scale * (0.20 + 0.80 * warmup_progress)
        actuator_guard = 1.0 - self.controller_actuator_guard_scale * long_priority * max(0.0, -value_pred_long) * max(0.15, 1.0 - retained_conflict)
        actuator_guard = max(0.35, actuator_guard)
        actuator = actuator_scale * actuator_guard * raw_actuator
        act_control_to_slow = actuator[:, 0:1]
        act_packet_to_slow = actuator[:, 1:2]
        act_carrier_to_slow = actuator[:, 2:3]
        act_control_to_packet = actuator[:, 3:4]
        act_release_gain = actuator[:, 4:5]
        act_release_threshold = actuator[:, 5:6]
        act_carrier_decay = actuator[:, 6:7]
        raw_topology_delta = torch.tanh(topology_state)
        if self.topology_open_only:
            raw_topology_delta = torch.relu(raw_topology_delta)
        topology_delta = self.topology_delta_scale * raw_topology_delta
        topo_control_to_packet = topology_delta[:, 0:1]
        topo_support_to_packet = topology_delta[:, min(1, self.topology_edges - 1):min(1, self.topology_edges - 1) + 1]
        topo_packet_to_carrier = topology_delta[:, min(2, self.topology_edges - 1):min(2, self.topology_edges - 1) + 1]
        topo_control_to_carrier = topology_delta[:, min(3, self.topology_edges - 1):min(3, self.topology_edges - 1) + 1]
        topo_carrier_to_slow = topology_delta[:, min(4, self.topology_edges - 1):min(4, self.topology_edges - 1) + 1]
        topo_packet_to_slow = topology_delta[:, min(5, self.topology_edges - 1):min(5, self.topology_edges - 1) + 1]

        eff_control_to_packet_scale = torch.clamp(
            self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open + act_control_to_packet + topo_control_to_packet,
            min=0.0,
        )
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open + topo_support_to_packet,
            min=0.0,
        )
        eff_control_to_slow_scale = torch.clamp(
            self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open + act_control_to_slow,
            min=0.0,
        )
        eff_packet_to_slow_scale = torch.clamp(
            self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open + act_packet_to_slow + topo_packet_to_slow,
            min=0.0,
        )
        eff_carrier_to_slow_scale = torch.clamp(
            self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open + act_carrier_to_slow + topo_carrier_to_slow,
            min=0.0,
        )
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open + act_release_gain,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open + act_release_threshold,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open + act_carrier_decay,
            min=0.0,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        eff_control_to_carrier_scale = torch.clamp(
            self.control_to_carrier_scale + topo_control_to_carrier,
            min=self.topology_carrier_floor,
        )
        eff_packet_to_carrier_scale = torch.clamp(
            self.packet_to_carrier_scale + topo_packet_to_carrier,
            min=self.topology_carrier_floor,
        )
        control_to_carrier = eff_control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = eff_packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )
        held_conflict = conflict_lane_activation.mean(dim=-1, keepdim=True)
        conflict_packet_drive = self.conflict_transform_scale * held_conflict * torch.tanh(
            self.conflict_to_packet(torch.tanh(conflict_potential))
        )
        conflict_carrier_drive = self.conflict_transform_scale * held_conflict * torch.tanh(
            self.conflict_to_carrier(torch.tanh(conflict_potential))
        )
        conflict_slow_drive = self.conflict_transform_scale * held_conflict * torch.tanh(
            self.conflict_to_slow(torch.tanh(conflict_potential))
        )
        conflict_transform_norm = (
            torch.norm(conflict_packet_drive)
            + torch.norm(conflict_carrier_drive)
            + torch.norm(conflict_slow_drive)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
            + conflict_packet_drive
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
            + conflict_carrier_drive
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + conflict_slow_drive
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
            + 0.08 * torch.tanh(self.controller_readout(new_controller_state))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        with torch.no_grad():
            route_activity = torch.tensor(
                [
                    float(control_to_slow.abs().mean().item()),
                    float(packet_to_slow.abs().mean().item()),
                    float(carrier_to_slow.abs().mean().item()),
                ],
                device=new_controller_state.device,
            )
            route_mean_now = float(route_activity.mean().item())
            route_balance = 1.0 - float(route_activity.std().item()) / (route_mean_now + 1e-6)
            route_balance = max(-1.0, min(1.0, route_balance))
            route_skew_now = max(0.0, 1.0 - route_balance)
            coherence_now = float(coherence_signal.mean().item())
            mismatch_now = float(state_mismatch.mean().item())
            release_now = float(release_strength.mean().item())
            actuator_norm_now = float(torch.norm(actuator).item())
            actuator_sat_now = float(raw_actuator.abs().mean().item())
            slow_normed = torch.tanh(new_slow)
            carrier_normed = carrier_readout
            packet_normed = packet_readout
            route_distribution = route_activity / (route_activity.sum() + 1e-6)
            route_entropy_now = float(
                (
                    -(route_distribution * torch.log(route_distribution + 1e-6)).sum()
                    / math.log(3.0)
                ).item()
            )
            carrier_anchor_now = float(
                (
                    (slow_normed * carrier_normed).sum(dim=-1)
                    / (slow_normed.norm(dim=-1) * carrier_normed.norm(dim=-1) + 1e-6)
                ).mean().item()
            )
            packet_anchor_now = float(
                (
                    (slow_normed * packet_normed).sum(dim=-1)
                    / (slow_normed.norm(dim=-1) * packet_normed.norm(dim=-1) + 1e-6)
                ).mean().item()
            )
            anchor_gap_now = abs(packet_anchor_now - carrier_anchor_now)
            conflict_drive_now = min(1.0, 0.40 * horizon_conflict + 0.30 * anchor_gap_now + 0.30 * route_entropy_now)
            conflict_containment_now = min(
                1.0,
                0.45 * max(0.0, route_skew_now - 0.35)
                + 0.35 * max(0.0, -value_pred_long)
                + 0.20 * max(0.0, 0.55 - route_entropy_now),
            )
            lane_drive_values = [
                horizon_conflict,
                route_skew_now,
                anchor_gap_now,
                max(0.0, packet_anchor_now - carrier_anchor_now),
                max(0.0, carrier_anchor_now - packet_anchor_now),
                max(0.0, 0.55 - route_entropy_now),
            ]
            if self.conflict_lanes > len(lane_drive_values):
                lane_drive_values.extend([conflict_drive_now] * (self.conflict_lanes - len(lane_drive_values)))
            lane_drive = torch.tensor(
                lane_drive_values[: self.conflict_lanes],
                device=conflict_potential.device,
                dtype=conflict_potential.dtype,
            ).view(1, -1)
            lane_containment_values = [
                max(0.0, -value_pred_long),
                max(0.0, route_skew_now - 0.35),
                max(0.0, anchor_gap_now - 0.35),
                max(0.0, packet_anchor_now - carrier_anchor_now),
                max(0.0, carrier_anchor_now - packet_anchor_now - 0.35),
                max(0.0, 0.55 - route_entropy_now),
            ]
            if self.conflict_lanes > len(lane_containment_values):
                lane_containment_values.extend([conflict_containment_now] * (self.conflict_lanes - len(lane_containment_values)))
            lane_containment = torch.tensor(
                lane_containment_values[: self.conflict_lanes],
                device=conflict_potential.device,
                dtype=conflict_potential.dtype,
            ).view(1, -1)
            conflict_transformation_now = min(
                1.0,
                float(conflict_transform_norm.item())
                / (self.conflict_transform_scale * math.sqrt(float(self.hidden_size + self.carrier_dim + self.packet_dim)) + 1e-6),
            )
            new_conflict_potential = torch.clamp(
                self.conflict_retention * conflict_potential
                + self.conflict_drive_scale * lane_drive
                - self.conflict_containment_scale * lane_containment,
                min=-2.0,
                max=2.0,
            )
            conflict_potential_now = float(torch.sigmoid(new_conflict_potential).mean().item())
            conflict_lane_strength = torch.softmax(new_conflict_potential.mean(dim=0), dim=0)
            conflict_lane_entropy_now = float(
                (
                    -(conflict_lane_strength * torch.log(conflict_lane_strength + 1e-6)).sum()
                    / math.log(float(self.conflict_lanes))
                ).item()
            ) if self.conflict_lanes > 1 else 0.0
            conflict_lane_dominance_now = float(conflict_lane_strength.max().item())
            route_boundary_pressure_now = float(anchor_gap_now * route_entropy_now)
            destructive_boundary_pressure_now = float(anchor_gap_now * route_skew_now)
            topology_drive_gain = max(0.0, route_boundary_pressure_now - destructive_boundary_pressure_now)
            topology_damping = min(0.25, 0.15 * destructive_boundary_pressure_now + 0.10 * conflict_containment_now)
            topology_drive = torch.tanh(self.conflict_to_topology(torch.tanh(new_conflict_potential)))
            new_topology_state = torch.clamp(
                (self.topology_retention - topology_damping) * topology_state
                + self.topology_update_scale * topology_drive_gain * topology_drive,
                min=-1.0,
                max=1.0,
            )
            topology_norm_now = float(torch.norm(new_topology_state).item())
            topology_delta_abs_now = float(torch.tanh(new_topology_state).abs().mean().item())

            self._controller_history.append(
                {
                    "controller_state": new_controller_state.detach().mean(dim=0, keepdim=True),
                    "actuator": actuator.detach().mean(dim=0, keepdim=True),
                    "actuator_norm": actuator_norm_now,
                    "actuator_sat": actuator_sat_now,
                    "value_pred": value_pred,
                    "value_pred_vector": value_pred_vector.detach().view(1, -1),
                    "coherence": coherence_now,
                    "mismatch": mismatch_now,
                    "release": release_now,
                    "route_mean": route_mean_now,
                    "route_balance": route_balance,
                    "route_skew": route_skew_now,
                    "route_entropy": route_entropy_now,
                    "carrier_anchor": carrier_anchor_now,
                    "packet_anchor": packet_anchor_now,
                    "conflict_potential": conflict_potential_now,
                    "anchor_gap": anchor_gap_now,
                    "conflict_transformation": conflict_transformation_now,
                    "conflict_lane_entropy": conflict_lane_entropy_now,
                    "conflict_lane_dominance": conflict_lane_dominance_now,
                    "route_boundary_pressure": route_boundary_pressure_now,
                    "destructive_boundary_pressure": destructive_boundary_pressure_now,
                    "topology_norm": topology_norm_now,
                    "topology_delta_abs": topology_delta_abs_now,
                }
            )
            max_keep = max(max(self.controller_horizon_delays) + self.controller_update_interval + 4, 32)
            if len(self._controller_history) > max_keep:
                self._controller_history = self._controller_history[-max_keep:]

            self.controller_step.add_(1.0)
            target = value_pred
            target_short = value_pred_short
            target_medium = value_pred_medium
            target_long = value_pred_long
            error = 0.0
            error_short = 0.0
            error_medium = 0.0
            error_long = 0.0
            if (
                not freeze_active
                and
                len(self._controller_history) > max(self.controller_horizon_delays)
                and int(self.controller_step.item()) % self.controller_update_interval == 0
            ):
                effective_warmup = min(1.0, max(0.0, float(self.controller_step.item()) / max(1, self.controller_warmup_steps)))
                horizon_targets: list[float] = []
                horizon_errors: list[float] = []
                actor_outer = torch.zeros_like(self.controller_actuator_head.weight.data)
                actor_bias_update = torch.zeros_like(self.controller_actuator_head.bias.data)
                value_weight_update = torch.zeros_like(self.controller_value_head.weight.data)
                value_bias_update = torch.zeros_like(self.controller_value_head.bias.data)

                for horizon_idx, (horizon_delay, horizon_weight) in enumerate(
                    zip(self.controller_horizon_delays, self.controller_horizon_weights)
                ):
                    past = self._controller_history[-(horizon_delay + 1)]
                    actuator_penalty = self.actuator_penalty_scale * (
                        0.65 * float(past["actuator_norm"]) / (self.actuator_delta_scale * math.sqrt(7.0) + 1e-6)
                        + 0.35 * float(past["actuator_sat"])
                    )
                    destructive_route_penalty = (
                        (0.08 + 0.02 * horizon_idx) * route_skew_now
                        + (0.06 + 0.03 * horizon_idx) * max(0.0, packet_anchor_now - carrier_anchor_now)
                    )
                    bounded_conflict_bonus = (
                        (0.06 + 0.04 * horizon_idx) * conflict_potential_now * route_entropy_now
                        - (0.05 + 0.03 * horizon_idx) * max(0.0, route_skew_now - 0.45)
                        + (0.04 + 0.02 * horizon_idx) * conflict_transformation_now
                    )
                    viability_delta = (
                        (0.34 - 0.04 * horizon_idx) * (coherence_now - float(past["coherence"]))
                        + (0.24 - 0.03 * horizon_idx) * (float(past["mismatch"]) - mismatch_now)
                        + (0.16 - 0.02 * horizon_idx) * (float(past["release"]) - release_now)
                        + (0.12 + 0.03 * horizon_idx) * (route_mean_now - float(past["route_mean"]))
                        + (0.08 + 0.03 * horizon_idx) * (route_balance - float(past["route_balance"]))
                        + (0.05 + 0.03 * horizon_idx) * (route_entropy_now - float(past["route_entropy"]))
                        + (0.10 + 0.04 * horizon_idx) * (carrier_anchor_now - float(past["carrier_anchor"]))
                        - (0.05 + 0.03 * horizon_idx) * (packet_anchor_now - float(past["packet_anchor"]))
                        + (0.07 + 0.05 * horizon_idx) * (conflict_potential_now - float(past["conflict_potential"]))
                        - (0.04 + 0.03 * horizon_idx) * max(0.0, anchor_gap_now - float(past["anchor_gap"]) - 0.15)
                        - actuator_penalty
                        - destructive_route_penalty
                        + bounded_conflict_bonus
                    )
                    target_i = math.tanh(2.3 * viability_delta + 0.20 * float(self.controller_bootstrap[horizon_idx].item()))
                    pred_i = float(past["value_pred_vector"][0, horizon_idx].item())
                    error_i = target_i - pred_i
                    horizon_targets.append(target_i)
                    horizon_errors.append(error_i)
                    past_state = past["controller_state"]
                    past_action = past["actuator"]
                    effective_horizon_weight = (
                        eff_short_weight if horizon_idx == 0 else
                        eff_medium_weight if horizon_idx == 1 else
                        eff_long_weight
                    )
                    value_weight_update[horizon_idx:horizon_idx + 1].add_(effective_horizon_weight * error_i * past_state)
                    value_bias_update[horizon_idx] += effective_horizon_weight * error_i
                    actor_outer.add_(effective_horizon_weight * error_i * (past_action.T @ past_state))
                    actor_bias_update.add_(effective_horizon_weight * error_i * past_action.view(-1))

                target_short, target_medium, target_long = horizon_targets
                error_short, error_medium, error_long = horizon_errors
                target = (
                    eff_short_weight * target_short
                    + eff_medium_weight * target_medium
                    + eff_long_weight * target_long
                )
                error = (
                    eff_short_weight * error_short
                    + eff_medium_weight * error_medium
                    + eff_long_weight * error_long
                )

                self.controller_value_head.weight.data.mul_(1.0 - self.controller_decay)
                self.controller_value_head.bias.data.mul_(1.0 - self.controller_decay)
                value_lr = self.value_eta * (0.20 + 0.80 * effective_warmup)
                self.controller_value_head.weight.data.add_(value_lr * value_weight_update)
                self.controller_value_head.bias.data.add_(value_lr * value_bias_update)

                self.controller_actuator_head.weight.data.mul_(1.0 - self.controller_decay)
                self.controller_actuator_head.bias.data.mul_(1.0 - self.controller_decay)
                actor_lr = self.controller_eta * (0.10 + 0.90 * effective_warmup)
                self.controller_actuator_head.weight.data.add_(actor_lr * actor_outer)
                self.controller_actuator_head.bias.data.add_(actor_lr * actor_bias_update)

                value_norm = float(torch.norm(self.controller_value_head.weight.data).item())
                if value_norm > self.controller_max_norm:
                    self.controller_value_head.weight.data.mul_(self.controller_max_norm / (value_norm + 1e-10))
                actor_norm = float(torch.norm(self.controller_actuator_head.weight.data).item())
                if actor_norm > self.controller_max_norm:
                    self.controller_actuator_head.weight.data.mul_(self.controller_max_norm / (actor_norm + 1e-10))
                self.controller_bootstrap.copy_(value_pred_vector.detach())

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
            "effective_control_to_carrier_scale_mean": float(eff_control_to_carrier_scale.mean().item()),
            "effective_packet_to_carrier_scale_mean": float(eff_packet_to_carrier_scale.mean().item()),
            "controller_state_norm": float(torch.norm(new_controller_state).item()),
            "controller_gate_mean": float(controller_gate.mean().item()),
            "controller_value_mean": value_pred,
            "controller_value_short_mean": value_pred_short,
            "controller_value_medium_mean": value_pred_medium,
            "controller_value_long_mean": value_pred_long,
            "controller_conflict_mean": float(horizon_conflict),
            "controller_long_priority_mean": float(long_priority),
            "controller_actuator_guard_mean": float(actuator_guard),
            "conflict_potential_mean": float(conflict_potential_now),
            "conflict_route_entropy_mean": float(route_entropy_now),
            "conflict_anchor_gap_mean": float(anchor_gap_now),
            "conflict_drive_mean": float(conflict_drive_now),
            "conflict_containment_mean": float(conflict_containment_now),
            "conflict_transformation_mean": float(conflict_transformation_now),
            "conflict_lane_entropy_mean": float(conflict_lane_entropy_now),
            "conflict_lane_dominance_mean": float(conflict_lane_dominance_now),
            "route_boundary_pressure_mean": float(route_boundary_pressure_now),
            "destructive_boundary_pressure_mean": float(destructive_boundary_pressure_now),
            "topology_state_norm": float(topology_norm_now),
            "topology_delta_abs_mean": float(topology_delta_abs_now),
            "topology_open_only_mean": float(self.topology_open_only),
            "topology_carrier_floor_mean": float(self.topology_carrier_floor),
            "topology_drive_gain_mean": float(topology_drive_gain),
            "topology_damping_mean": float(topology_damping),
            "topology_control_packet_delta_mean": float(torch.tanh(new_topology_state[:, 0]).mean().item()),
            "topology_support_packet_delta_mean": float(torch.tanh(new_topology_state[:, min(1, self.topology_edges - 1)]).mean().item()),
            "topology_packet_carrier_delta_mean": float(torch.tanh(new_topology_state[:, min(2, self.topology_edges - 1)]).mean().item()),
            "topology_control_carrier_delta_mean": float(torch.tanh(new_topology_state[:, min(3, self.topology_edges - 1)]).mean().item()),
            "topology_carrier_slow_delta_mean": float(torch.tanh(new_topology_state[:, min(4, self.topology_edges - 1)]).mean().item()),
            "topology_packet_slow_delta_mean": float(torch.tanh(new_topology_state[:, min(5, self.topology_edges - 1)]).mean().item()),
            "conflict_horizon_lane_mean": float(torch.sigmoid(new_conflict_potential[:, 0]).mean().item()),
            "conflict_route_lane_mean": float(torch.sigmoid(new_conflict_potential[:, min(1, self.conflict_lanes - 1)]).mean().item()),
            "conflict_anchor_lane_mean": float(torch.sigmoid(new_conflict_potential[:, min(2, self.conflict_lanes - 1)]).mean().item()),
            "conflict_transform_norm": float(conflict_transform_norm.item()),
            "conflict_packet_drive_norm": float(torch.norm(conflict_packet_drive).item()),
            "conflict_carrier_drive_norm": float(torch.norm(conflict_carrier_drive).item()),
            "conflict_slow_drive_norm": float(torch.norm(conflict_slow_drive).item()),
            "conflict_resolution_mean": float(conflict_containment_now),
            "controller_target_mean": float(target),
            "controller_target_short_mean": float(target_short),
            "controller_target_medium_mean": float(target_medium),
            "controller_target_long_mean": float(target_long),
            "controller_error_mean": float(error),
            "controller_error_short_mean": float(error_short),
            "controller_error_medium_mean": float(error_medium),
            "controller_error_long_mean": float(error_long),
            "controller_actuator_norm": float(torch.norm(actuator).item()),
            "controller_warmup_progress_mean": float(warmup_progress),
            "controller_freeze_active_mean": float(freeze_active),
            "route_balance_mean": float(route_balance),
            "route_skew_mean": float(route_skew_now),
            "carrier_anchor_mean": float(carrier_anchor_now),
            "packet_anchor_mean": float(packet_anchor_now),
            "actuator_control_to_slow_mean": float(act_control_to_slow.mean().item()),
            "actuator_packet_to_slow_mean": float(act_packet_to_slow.mean().item()),
            "actuator_carrier_to_slow_mean": float(act_carrier_to_slow.mean().item()),
            "actuator_control_to_packet_mean": float(act_control_to_packet.mean().item()),
            "actuator_release_gain_mean": float(act_release_gain.mean().item()),
            "actuator_release_threshold_mean": float(act_release_threshold.mean().item()),
            "actuator_carrier_decay_mean": float(act_carrier_decay.mean().item()),
            "controller_value_weight_norm": float(torch.norm(self.controller_value_head.weight).item()),
            "controller_actuator_weight_norm": float(torch.norm(self.controller_actuator_head.weight).item()),
        }
        return (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
        )


class DemianNativeV7Substrate(DemianNativeV6Substrate):
    """Native v7: explicit temporal-self and boundary organs on top of v6."""

    def __init__(
        self,
        hidden_size: int,
        self_retention: float = 0.94,
        active_retention: float = 0.72,
        projection_retention: float = 0.88,
        boundary_retention: float = 0.90,
        self_injection_scale: float = 0.035,
        boundary_injection_scale: float = 0.025,
        projection_topology_scale: float = 0.050,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.self_retention = self_retention
        self.active_retention = active_retention
        self.projection_retention = projection_retention
        self.boundary_retention = boundary_retention
        self.self_injection_scale = self_injection_scale
        self.boundary_injection_scale = boundary_injection_scale
        self.projection_topology_scale = projection_topology_scale

        temporal_in_dim = hidden_size * 10
        boundary_in_dim = hidden_size * 6
        self.ancestry_mix = nn.Linear(temporal_in_dim, hidden_size)
        self.active_mix = nn.Linear(temporal_in_dim, hidden_size)
        self.projection_mix = nn.Linear(temporal_in_dim, hidden_size)
        self.boundary_mix = nn.Linear(boundary_in_dim, hidden_size)
        self.boundary_gate = nn.Linear(boundary_in_dim, hidden_size)
        self.boundary_action_head = nn.Linear(hidden_size, 5)

        self.ancestry_to_carrier = nn.Linear(hidden_size, self.carrier_dim, bias=False)
        self.projection_to_carrier = nn.Linear(hidden_size, self.carrier_dim, bias=False)
        self.active_to_packet = nn.Linear(hidden_size, self.packet_dim, bias=False)
        self.boundary_to_packet = nn.Linear(hidden_size, self.packet_dim, bias=False)
        self.boundary_to_control = nn.Linear(hidden_size, self.control_dim, bias=False)
        self.projection_to_control = nn.Linear(hidden_size, self.control_dim, bias=False)
        self.projection_to_topology = nn.Linear(hidden_size, self.topology_edges, bias=False)
        self.v7_expose_gate = nn.Linear(hidden_size * 5, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        base = super().initial_state(batch_size, device)
        ancestry_state = torch.zeros(batch_size, self.hidden_size, device=device)
        active_state = torch.zeros(batch_size, self.hidden_size, device=device)
        projection_state = torch.zeros(batch_size, self.hidden_size, device=device)
        boundary_state = torch.zeros(batch_size, self.hidden_size, device=device)
        return (*base, ancestry_state, active_state, projection_state, boundary_state)

    def _split_v7_state(self, state: tuple[torch.Tensor, ...]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[:11], state[11], state[12], state[13], state[14]

    def state_components(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        if len(state) == 11:
            return super().state_components(state)
        base_state, ancestry_state, active_state, projection_state, boundary_state = self._split_v7_state(state)
        components = super().state_components(base_state)
        components.update(
            {
                "ancestry": ancestry_state,
                "active_self": active_state,
                "projection": projection_state,
                "boundary": boundary_state,
            }
        )
        return components

    def state_vector(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(state) == 11:
            return super().state_vector(state)
        base_state, ancestry_state, active_state, projection_state, boundary_state = self._split_v7_state(state)
        base_view = super().state_vector(base_state)
        ancestry_view = 0.10 * torch.tanh(ancestry_state)
        active_view = 0.08 * torch.tanh(active_state)
        projection_view = 0.08 * torch.tanh(projection_state)
        boundary_view = 0.06 * torch.tanh(boundary_state)
        gate = torch.sigmoid(
            self.v7_expose_gate(
                torch.cat([base_view, ancestry_view, active_view, projection_view, boundary_view], dim=-1)
            )
        )
        anchored = base_view + ancestry_view + active_view + projection_view - 0.5 * boundary_view
        return gate * base_view + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        base_state, ancestry_state, active_state, projection_state, boundary_state = self._split_v7_state(state)
        (
            fast,
            slow,
            long_carrier,
            short_support,
            packet,
            control_short,
            control_long,
            tightness,
            controller_state,
            conflict_potential,
            topology_state,
        ) = base_state

        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_view = torch.tanh(self.packet_readout(packet))
        controller_view = torch.tanh(self.controller_readout(controller_state))
        conflict_view = torch.tanh(self.conflict_readout(torch.tanh(conflict_potential)))
        topology_view = torch.tanh(self.topology_readout(torch.tanh(topology_state)))
        temporal_features = torch.cat(
            [
                torch.tanh(fast),
                slow_view,
                carrier_view,
                packet_view,
                controller_view,
                conflict_view,
                topology_view,
                torch.tanh(ancestry_state),
                torch.tanh(active_state),
                torch.tanh(projection_state),
            ],
            dim=-1,
        )

        boundary_logits = self.boundary_action_head(torch.tanh(boundary_state))
        boundary_actions = torch.sigmoid(boundary_logits)
        internalize = boundary_actions[:, 0:1]
        reject = boundary_actions[:, 1:2]
        quarantine = boundary_actions[:, 2:3]
        transmit = boundary_actions[:, 3:4]
        transform = boundary_actions[:, 4:5]
        permeability = torch.clamp(0.55 * internalize + 0.35 * transmit + 0.25 * transform - 0.40 * reject, 0.0, 1.0)

        adjusted_fast = fast + self.self_injection_scale * (
            0.70 * torch.tanh(active_state)
            + 0.45 * torch.tanh(projection_state)
            - 0.40 * reject * torch.tanh(boundary_state)
        )
        adjusted_slow = slow + self.self_injection_scale * internalize * torch.tanh(ancestry_state)
        adjusted_carrier = long_carrier + self.self_injection_scale * (
            internalize * torch.tanh(self.ancestry_to_carrier(ancestry_state))
            + transmit * torch.tanh(self.projection_to_carrier(projection_state))
        )
        adjusted_packet = packet + self.boundary_injection_scale * (
            transform * torch.tanh(self.active_to_packet(active_state))
            - quarantine * torch.tanh(self.boundary_to_packet(boundary_state))
        )
        adjusted_control_short = control_short + self.boundary_injection_scale * torch.tanh(self.boundary_to_control(boundary_state))
        adjusted_control_long = control_long + self.boundary_injection_scale * torch.tanh(self.projection_to_control(projection_state))
        adjusted_tightness = tightness + 0.02 * (internalize - reject).mean(dim=-1, keepdim=True)

        new_base = super().step(
            (
                adjusted_fast,
                adjusted_slow,
                adjusted_carrier,
                short_support,
                adjusted_packet,
                adjusted_control_short,
                adjusted_control_long,
                adjusted_tightness,
                controller_state,
                conflict_potential,
                topology_state,
            )
        )
        (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
        ) = new_base

        new_controller_view = torch.tanh(self.controller_readout(new_controller_state))
        new_conflict_view = torch.tanh(self.conflict_readout(torch.tanh(new_conflict_potential)))
        new_topology_view = torch.tanh(self.topology_readout(torch.tanh(new_topology_state)))
        new_slow_view = torch.tanh(self.slow_readout(new_slow))
        new_carrier_view = torch.tanh(self.long_carrier_readout(new_long_carrier))
        new_packet_view = torch.tanh(self.packet_readout(new_packet))
        next_temporal_features = torch.cat(
            [
                torch.tanh(new_fast),
                new_slow_view,
                new_carrier_view,
                new_packet_view,
                new_controller_view,
                new_conflict_view,
                new_topology_view,
                torch.tanh(ancestry_state),
                torch.tanh(active_state),
                torch.tanh(projection_state),
            ],
            dim=-1,
        )
        ancestry_candidate = torch.tanh(self.ancestry_mix(next_temporal_features))
        active_candidate = torch.tanh(self.active_mix(next_temporal_features))
        projection_candidate = torch.tanh(self.projection_mix(next_temporal_features))
        new_ancestry_state = (
            self.self_retention * ancestry_state
            + (1.0 - self.self_retention) * ancestry_candidate
            + 0.015 * internalize * new_slow_view
            + 0.010 * transmit * new_carrier_view
        )
        new_active_state = self.active_retention * active_state + (1.0 - self.active_retention) * active_candidate
        new_projection_state = (
            self.projection_retention * projection_state
            + (1.0 - self.projection_retention) * projection_candidate
            + 0.050 * torch.tanh(new_active_state)
            + 0.015 * transform * new_topology_view
        )

        boundary_features = torch.cat(
            [
                new_active_state.tanh(),
                new_projection_state.tanh(),
                new_ancestry_state.tanh(),
                new_slow_view,
                new_conflict_view,
                new_topology_view,
            ],
            dim=-1,
        )
        boundary_gate = torch.sigmoid(self.boundary_gate(boundary_features))
        boundary_candidate = torch.tanh(self.boundary_mix(boundary_features))
        new_boundary_state = (
            self.boundary_retention * boundary_state
            + (1.0 - self.boundary_retention) * boundary_gate * boundary_candidate
            + 0.010 * (quarantine - internalize) * new_conflict_view
        )
        new_boundary_actions = torch.sigmoid(self.boundary_action_head(torch.tanh(new_boundary_state)))
        new_internalize = new_boundary_actions[:, 0:1]
        new_reject = new_boundary_actions[:, 1:2]
        new_quarantine = new_boundary_actions[:, 2:3]
        new_transmit = new_boundary_actions[:, 3:4]
        new_transform = new_boundary_actions[:, 4:5]
        new_permeability = torch.clamp(
            0.55 * new_internalize + 0.35 * new_transmit + 0.25 * new_transform - 0.40 * new_reject,
            0.0,
            1.0,
        )
        topology_projection = self.projection_topology_scale * new_permeability * torch.tanh(
            self.projection_to_topology(new_projection_state)
        )
        new_topology_state = torch.clamp(new_topology_state + topology_projection, min=-1.0, max=1.0)

        ancestry_normed = torch.tanh(new_ancestry_state)
        active_normed = torch.tanh(new_active_state)
        projection_normed = torch.tanh(new_projection_state)
        boundary_normed = torch.tanh(new_boundary_state)
        continuity_alignment = torch.nn.functional.cosine_similarity(ancestry_normed, active_normed, dim=-1).mean()
        prospective_alignment = torch.nn.functional.cosine_similarity(active_normed, projection_normed, dim=-1).mean()
        boundary_alignment = torch.nn.functional.cosine_similarity(boundary_normed, new_conflict_view, dim=-1).mean()
        ancestry_projection_alignment = torch.nn.functional.cosine_similarity(ancestry_normed, projection_normed, dim=-1).mean()

        self._step_aux.update(
            {
                "v7_ancestry_state_norm": float(torch.norm(new_ancestry_state).item()),
                "v7_active_state_norm": float(torch.norm(new_active_state).item()),
                "v7_projection_state_norm": float(torch.norm(new_projection_state).item()),
                "v7_boundary_state_norm": float(torch.norm(new_boundary_state).item()),
                "v7_continuity_alignment_mean": float(continuity_alignment.item()),
                "v7_prospective_alignment_mean": float(prospective_alignment.item()),
                "v7_ancestry_projection_alignment_mean": float(ancestry_projection_alignment.item()),
                "v7_boundary_conflict_alignment_mean": float(boundary_alignment.item()),
                "v7_boundary_internalize_mean": float(new_internalize.mean().item()),
                "v7_boundary_reject_mean": float(new_reject.mean().item()),
                "v7_boundary_quarantine_mean": float(new_quarantine.mean().item()),
                "v7_boundary_transmit_mean": float(new_transmit.mean().item()),
                "v7_boundary_transform_mean": float(new_transform.mean().item()),
                "v7_boundary_permeability_mean": float(new_permeability.mean().item()),
                "v7_boundary_gate_mean": float(boundary_gate.mean().item()),
                "v7_projection_topology_norm": float(torch.norm(topology_projection).item()),
                "v7_self_injection_norm": float(torch.norm(adjusted_fast - fast).item()),
                "v7_boundary_packet_injection_norm": float(torch.norm(adjusted_packet - packet).item()),
            }
        )
        return (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
            new_ancestry_state,
            new_active_state,
            new_projection_state,
            new_boundary_state,
        )


class DemianNativeV71Substrate(DemianNativeV7Substrate):
    """Native v7.1: v7 plus compressed route-transition trajectory memory."""

    def __init__(
        self,
        hidden_size: int,
        trajectory_retention: float = 0.93,
        valence_retention: float = 0.91,
        reuse_retention: float = 0.89,
        trajectory_injection_scale: float = 0.028,
        trajectory_projection_scale: float = 0.045,
        trajectory_boundary_scale: float = 0.020,
        trajectory_topology_scale: float = 0.025,
        trajectory_carrier_norm_floor: float = 48.0,
        trajectory_carrier_norm_cap: float = 96.0,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.trajectory_retention = trajectory_retention
        self.valence_retention = valence_retention
        self.reuse_retention = reuse_retention
        self.trajectory_injection_scale = trajectory_injection_scale
        self.trajectory_projection_scale = trajectory_projection_scale
        self.trajectory_boundary_scale = trajectory_boundary_scale
        self.trajectory_topology_scale = trajectory_topology_scale
        self.trajectory_carrier_norm_floor = trajectory_carrier_norm_floor
        self.trajectory_carrier_norm_cap = trajectory_carrier_norm_cap

        transition_in_dim = hidden_size * 9
        self.transition_memory_mix = nn.Linear(transition_in_dim, hidden_size)
        self.transition_valence_mix = nn.Linear(transition_in_dim, hidden_size)
        self.transition_reuse_mix = nn.Linear(transition_in_dim, hidden_size)
        self.trajectory_to_ancestry = nn.Linear(hidden_size, hidden_size, bias=False)
        self.trajectory_to_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.trajectory_to_boundary = nn.Linear(hidden_size, hidden_size, bias=False)
        self.trajectory_to_topology = nn.Linear(hidden_size, self.topology_edges, bias=False)
        self.v71_expose_gate = nn.Linear(hidden_size * 5, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        base = super().initial_state(batch_size, device)
        trajectory_memory = torch.zeros(batch_size, self.hidden_size, device=device)
        trajectory_valence = torch.zeros(batch_size, self.hidden_size, device=device)
        trajectory_reuse = torch.zeros(batch_size, self.hidden_size, device=device)
        previous_surface = torch.zeros(batch_size, self.hidden_size, device=device)
        return (*base, trajectory_memory, trajectory_valence, trajectory_reuse, previous_surface)

    def _split_v71_state(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[:15], state[15], state[16], state[17], state[18]

    def state_components(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        if len(state) == 11 or len(state) == 15:
            return super().state_components(state)
        v7_state, trajectory_memory, trajectory_valence, trajectory_reuse, previous_surface = self._split_v71_state(state)
        components = super().state_components(v7_state)
        components.update(
            {
                "trajectory_memory": trajectory_memory,
                "trajectory_valence": trajectory_valence,
                "trajectory_reuse": trajectory_reuse,
                "previous_surface": previous_surface,
            }
        )
        return components

    def state_vector(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(state) == 11 or len(state) == 15:
            return super().state_vector(state)
        v7_state, trajectory_memory, trajectory_valence, trajectory_reuse, _previous_surface = self._split_v71_state(state)
        base_view = super().state_vector(v7_state)
        memory_view = 0.08 * torch.tanh(trajectory_memory)
        valence_view = 0.05 * torch.tanh(trajectory_valence)
        reuse_view = 0.05 * torch.tanh(trajectory_reuse)
        gate = torch.sigmoid(
            self.v71_expose_gate(
                torch.cat([base_view, memory_view, valence_view, reuse_view, torch.tanh(base_view - memory_view)], dim=-1)
            )
        )
        anchored = base_view + memory_view + valence_view + reuse_view
        return gate * base_view + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        v7_state, trajectory_memory, trajectory_valence, trajectory_reuse, previous_surface = self._split_v71_state(state)
        (
            fast,
            slow,
            long_carrier,
            short_support,
            packet,
            control_short,
            control_long,
            tightness,
            controller_state,
            conflict_potential,
            topology_state,
            ancestry_state,
            active_state,
            projection_state,
            boundary_state,
        ) = v7_state

        memory_drive = torch.tanh(trajectory_memory + 0.65 * trajectory_valence + 0.45 * trajectory_reuse)
        adjusted_v7_state = (
            fast + self.trajectory_injection_scale * memory_drive,
            slow,
            long_carrier,
            short_support,
            packet,
            control_short,
            control_long,
            tightness,
            controller_state,
            conflict_potential,
            topology_state,
            ancestry_state + self.trajectory_injection_scale * torch.tanh(self.trajectory_to_ancestry(memory_drive)),
            active_state,
            projection_state + self.trajectory_projection_scale * torch.tanh(self.trajectory_to_projection(memory_drive)),
            boundary_state + self.trajectory_boundary_scale * torch.tanh(self.trajectory_to_boundary(memory_drive)),
        )

        old_surface = super().state_vector(v7_state)
        old_conflict_view = torch.tanh(self.conflict_readout(torch.tanh(conflict_potential)))
        old_topology_view = torch.tanh(self.topology_readout(torch.tanh(topology_state)))
        old_boundary_normed = torch.tanh(boundary_state)
        new_v7_state = super().step(adjusted_v7_state)
        (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
            new_ancestry_state,
            new_active_state,
            new_projection_state,
            new_boundary_state,
        ) = new_v7_state
        boundary_actions = torch.sigmoid(self.boundary_action_head(torch.tanh(new_boundary_state)))
        boundary_internalize = boundary_actions[:, 0:1]
        boundary_reject = boundary_actions[:, 1:2]
        boundary_transmit = boundary_actions[:, 3:4]
        boundary_transform = boundary_actions[:, 4:5]
        boundary_permeability = torch.clamp(
            0.55 * boundary_internalize
            + 0.35 * boundary_transmit
            + 0.25 * boundary_transform
            - 0.40 * boundary_reject,
            min=0.0,
            max=1.0,
        )
        prior_valence_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(trajectory_memory + trajectory_reuse),
            torch.tanh(trajectory_valence),
            dim=-1,
        ).view(-1, 1)
        prior_valence_open = torch.clamp(0.5 + 0.5 * prior_valence_alignment, min=0.0, max=1.0)
        carrier_cap_open = torch.clamp(0.55 * boundary_permeability + 0.45 * prior_valence_open, min=0.0, max=1.0)
        effective_carrier_cap = self.trajectory_carrier_norm_floor + (
            self.trajectory_carrier_norm_cap - self.trajectory_carrier_norm_floor
        ) * carrier_cap_open
        carrier_norm = torch.norm(new_long_carrier, dim=-1, keepdim=True) / math.sqrt(max(self.carrier_dim, 1))
        carrier_guard_scale = torch.clamp(effective_carrier_cap / (carrier_norm + 1e-6), max=1.0)
        new_long_carrier = new_long_carrier * carrier_guard_scale
        new_v7_state = (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
            new_ancestry_state,
            new_active_state,
            new_projection_state,
            new_boundary_state,
        )

        new_surface = super().state_vector(new_v7_state)
        surface_delta = torch.tanh(new_surface - old_surface)
        previous_delta = torch.tanh(old_surface - previous_surface)
        new_conflict_view = torch.tanh(self.conflict_readout(torch.tanh(new_conflict_potential)))
        new_topology_view = torch.tanh(self.topology_readout(torch.tanh(new_topology_state)))
        new_boundary_normed = torch.tanh(new_boundary_state)
        conflict_delta = torch.tanh(new_conflict_view - old_conflict_view)
        topology_delta = torch.tanh(new_topology_view - old_topology_view)
        boundary_delta = torch.tanh(new_boundary_normed - old_boundary_normed)
        continuity_delta = torch.tanh(torch.tanh(new_active_state) - torch.tanh(new_ancestry_state))
        projection_delta = torch.tanh(torch.tanh(new_projection_state) - torch.tanh(new_active_state))

        transition_features = torch.cat(
            [
                torch.tanh(old_surface),
                torch.tanh(new_surface),
                surface_delta,
                previous_delta,
                conflict_delta,
                topology_delta,
                boundary_delta,
                continuity_delta,
                projection_delta,
            ],
            dim=-1,
        )
        transition_candidate = torch.tanh(self.transition_memory_mix(transition_features))
        valence_candidate = torch.tanh(self.transition_valence_mix(transition_features))
        reuse_candidate = torch.tanh(self.transition_reuse_mix(transition_features))
        transition_reuse_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(trajectory_memory),
            transition_candidate,
            dim=-1,
        ).view(-1, 1)
        transition_valence_signal = torch.clamp(
            0.45 * transition_reuse_alignment
            + 0.35 * torch.nn.functional.cosine_similarity(torch.tanh(new_ancestry_state), torch.tanh(new_active_state), dim=-1).view(-1, 1)
            + 0.20 * torch.nn.functional.cosine_similarity(torch.tanh(new_active_state), torch.tanh(new_projection_state), dim=-1).view(-1, 1),
            min=-1.0,
            max=1.0,
        )
        new_trajectory_memory = (
            self.trajectory_retention * trajectory_memory
            + (1.0 - self.trajectory_retention) * transition_candidate
            + 0.012 * transition_valence_signal * surface_delta
        )
        new_trajectory_valence = (
            self.valence_retention * trajectory_valence
            + (1.0 - self.valence_retention) * valence_candidate
            + 0.010 * transition_valence_signal * torch.tanh(new_surface)
        )
        new_trajectory_reuse = (
            self.reuse_retention * trajectory_reuse
            + (1.0 - self.reuse_retention) * reuse_candidate
            + 0.012 * transition_reuse_alignment * transition_candidate
        )
        trajectory_projection_bridge = (
            self.trajectory_projection_scale
            * torch.clamp(transition_valence_signal, min=0.0)
            * (
                0.35 * torch.tanh(self.trajectory_to_projection(torch.tanh(new_trajectory_memory)))
                + 0.65 * torch.tanh(new_trajectory_memory)
            )
        )
        pre_bridge_projection = new_projection_state
        new_projection_state = new_projection_state + trajectory_projection_bridge
        projection_alignment_after_bridge = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_projection_state),
            dim=-1,
        ).view(-1, 1)
        projection_correction = (
            8.0 * self.trajectory_projection_scale
            * torch.relu(-projection_alignment_after_bridge)
            * torch.tanh(new_trajectory_memory)
        )
        new_projection_state = new_projection_state + projection_correction
        projection_alignment_after_correction = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_projection_state),
            dim=-1,
        ).view(-1, 1)
        projection_norm_for_correction = torch.norm(torch.tanh(new_projection_state), dim=-1, keepdim=True)
        projection_second_correction = (
            0.75
            * torch.relu(-projection_alignment_after_correction)
            * projection_norm_for_correction
            * torch.tanh(new_trajectory_memory)
        )
        new_projection_state = new_projection_state + projection_second_correction
        projection_alignment_after_second = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_projection_state),
            dim=-1,
        ).view(-1, 1)
        projection_final_correction = (
            0.50
            * torch.relu(-projection_alignment_after_second)
            * torch.norm(torch.tanh(new_projection_state), dim=-1, keepdim=True)
            * torch.tanh(new_trajectory_memory)
        )
        new_projection_state = new_projection_state + projection_final_correction
        trajectory_topology = self.trajectory_topology_scale * torch.clamp(transition_valence_signal, min=0.0) * torch.tanh(
            self.trajectory_to_topology(torch.tanh(new_trajectory_memory))
        )
        new_topology_state = torch.clamp(new_topology_state + trajectory_topology, min=-1.0, max=1.0)
        trajectory_boundary_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_boundary_state),
            dim=-1,
        ).mean()
        trajectory_projection_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_projection_state),
            dim=-1,
        ).mean()

        self._step_aux.update(
            {
                "v71_trajectory_memory_norm": float(torch.norm(new_trajectory_memory).item()),
                "v71_trajectory_valence_norm": float(torch.norm(new_trajectory_valence).item()),
                "v71_trajectory_reuse_norm": float(torch.norm(new_trajectory_reuse).item()),
                "v71_surface_delta_norm": float(torch.norm(surface_delta).item()),
                "v71_previous_delta_alignment_mean": float(
                    torch.nn.functional.cosine_similarity(surface_delta, previous_delta, dim=-1).mean().item()
                ),
                "v71_transition_reuse_alignment_mean": float(transition_reuse_alignment.mean().item()),
                "v71_transition_valence_signal_mean": float(transition_valence_signal.mean().item()),
                "v71_trajectory_boundary_alignment_mean": float(trajectory_boundary_alignment.item()),
                "v71_trajectory_projection_alignment_mean": float(trajectory_projection_alignment.item()),
                "v71_trajectory_projection_bridge_norm": float(torch.norm(trajectory_projection_bridge + projection_correction + projection_second_correction + projection_final_correction).item()),
                "v71_projection_correction_norm": float(torch.norm(projection_correction + projection_second_correction + projection_final_correction).item()),
                "v71_projection_second_correction_norm": float(torch.norm(projection_second_correction).item()),
                "v71_projection_final_correction_norm": float(torch.norm(projection_final_correction).item()),
                "v71_projection_pre_bridge_gap_norm": float(torch.norm(pre_bridge_projection - new_projection_state).item()),
                "v71_trajectory_topology_norm": float(torch.norm(trajectory_topology).item()),
                "v71_carrier_guard_scale_mean": float(carrier_guard_scale.mean().item()),
                "v71_effective_carrier_cap_mean": float(effective_carrier_cap.mean().item()),
                "v71_carrier_cap_open_mean": float(carrier_cap_open.mean().item()),
                "v71_prior_valence_open_mean": float(prior_valence_open.mean().item()),
                "v71_carrier_norm_capped_mean": float(
                    (torch.norm(new_long_carrier, dim=-1) / math.sqrt(max(self.carrier_dim, 1))).mean().item()
                ),
                "topology_state_norm": float(torch.norm(new_topology_state).item()),
            }
        )
        return (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
            new_ancestry_state,
            new_active_state,
            new_projection_state,
            new_boundary_state,
            new_trajectory_memory,
            new_trajectory_valence,
            new_trajectory_reuse,
            new_surface.detach(),
        )


class DemianNativeV72Substrate(DemianNativeV71Substrate):
    """Native v7.2: v7.1 plus constrained-environment metabolic resource."""

    def __init__(
        self,
        hidden_size: int,
        resource_retention: float = 0.94,
        resource_gain: float = 0.08,
        resource_cost_scale: float = 0.07,
        environment_constraint_scale: float = 0.10,
        anti_self_scale: float = 0.18,
        rss_floor: float = 0.08,
        rss_target: float = 0.32,
        resource_tightness_scale: float = 0.03,
        v72_exposure_scale: float = 0.04,
        refractory_surface_delta_threshold: Optional[float] = None,
        refractory_resource_gain_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.resource_retention = resource_retention
        self.resource_gain = resource_gain
        self.resource_cost_scale = resource_cost_scale
        self.environment_constraint_scale = environment_constraint_scale
        self.anti_self_scale = anti_self_scale
        self.rss_floor = rss_floor
        self.rss_target = rss_target
        self.resource_tightness_scale = resource_tightness_scale
        self.v72_exposure_scale = v72_exposure_scale
        self.refractory_surface_delta_threshold = refractory_surface_delta_threshold
        self.refractory_resource_gain_scale = refractory_resource_gain_scale
        self.resource_to_fast = nn.Linear(1, hidden_size, bias=False)
        self.resource_to_boundary = nn.Linear(1, hidden_size, bias=False)
        self.resource_to_projection = nn.Linear(1, hidden_size, bias=False)
        self.v72_expose_gate = nn.Linear(hidden_size * 3 + 1, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        base = super().initial_state(batch_size, device)
        resource_state = torch.zeros(batch_size, 1, device=device)
        return (*base, resource_state)

    def _split_v72_state(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        return state[:19], state[19]

    def state_components(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        if len(state) <= 19:
            return super().state_components(state)
        v71_state, resource_state = self._split_v72_state(state)
        components = super().state_components(v71_state)
        components["resource"] = resource_state
        return components

    def state_vector(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(state) <= 19:
            return super().state_vector(state)
        v71_state, resource_state = self._split_v72_state(state)
        base_view = super().state_vector(v71_state)
        resource_open = torch.sigmoid(resource_state)
        resource_view = torch.tanh(self.resource_to_fast(resource_state))
        gate = torch.sigmoid(
            self.v72_expose_gate(
                torch.cat(
                    [
                        base_view,
                        resource_view,
                        torch.tanh(base_view - resource_view),
                        resource_open,
                    ],
                    dim=-1,
                )
            )
        )
        exposed = base_view + self.v72_exposure_scale * resource_open * resource_view
        return gate * base_view + (1.0 - gate) * exposed

    def step(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        v71_state, resource_state = self._split_v72_state(state)
        (
            fast,
            slow,
            long_carrier,
            short_support,
            packet,
            control_short,
            control_long,
            tightness,
            controller_state,
            conflict_potential,
            topology_state,
            ancestry_state,
            active_state,
            projection_state,
            boundary_state,
            trajectory_memory,
            trajectory_valence,
            trajectory_reuse,
            previous_surface,
        ) = v71_state

        resource_open = torch.sigmoid(resource_state)
        scarcity = 1.0 - resource_open
        boundary_actions = torch.sigmoid(self.boundary_action_head(torch.tanh(boundary_state)))
        internalize = boundary_actions[:, 0:1]
        reject = boundary_actions[:, 1:2]
        quarantine = boundary_actions[:, 2:3]
        transmit = boundary_actions[:, 3:4]
        transform = boundary_actions[:, 4:5]
        permeability = torch.clamp(0.55 * internalize + 0.35 * transmit + 0.25 * transform - 0.40 * reject, 0.0, 1.0)
        self_view = torch.tanh(active_state + trajectory_memory)
        external_view = torch.tanh(projection_state + boundary_state)
        anti_self_pressure = torch.relu(-torch.nn.functional.cosine_similarity(self_view, external_view, dim=-1)).view(-1, 1)

        current_surface = super().state_vector(v71_state)
        surface_discontinuity = torch.norm(
            torch.tanh(current_surface - previous_surface),
            dim=-1,
            keepdim=True,
        ) / math.sqrt(max(self.hidden_size, 1))
        if self.refractory_surface_delta_threshold is None:
            resource_authority = torch.ones_like(resource_open)
        else:
            refractory_active = (
                surface_discontinuity > self.refractory_surface_delta_threshold
            ).to(resource_open.dtype)
            resource_authority = 1.0 - refractory_active * (
                1.0 - self.refractory_resource_gain_scale
            )
        effective_resource_gain = self.resource_gain * resource_authority

        constraint = self.environment_constraint_scale * scarcity
        constrained_v71_state = (
            fast
            + effective_resource_gain * resource_open * torch.tanh(self.resource_to_fast(resource_state))
            - self.anti_self_scale * anti_self_pressure * torch.tanh(boundary_state),
            slow,
            long_carrier * (1.0 - 0.20 * constraint),
            short_support,
            packet * (1.0 - constraint) - 0.05 * anti_self_pressure * torch.tanh(packet),
            control_short,
            control_long,
            tightness + self.resource_tightness_scale * resource_open - 0.02 * anti_self_pressure,
            controller_state,
            conflict_potential + 0.04 * anti_self_pressure * torch.tanh(conflict_potential + 1.0),
            topology_state,
            ancestry_state,
            active_state,
            projection_state
            + effective_resource_gain * resource_open * torch.tanh(self.resource_to_projection(resource_state))
            - 0.04 * anti_self_pressure * torch.tanh(projection_state),
            boundary_state
            + effective_resource_gain * (resource_open - 0.5) * torch.tanh(self.resource_to_boundary(resource_state))
            + 0.05 * anti_self_pressure * torch.tanh(boundary_state),
            trajectory_memory,
            trajectory_valence,
            trajectory_reuse,
            previous_surface,
        )

        new_v71_state = super().step(constrained_v71_state)
        (
            new_fast,
            new_slow,
            new_long_carrier,
            new_short_support,
            new_packet,
            new_control_short,
            new_control_long,
            new_tightness,
            new_controller_state,
            new_conflict_potential,
            new_topology_state,
            new_ancestry_state,
            new_active_state,
            new_projection_state,
            new_boundary_state,
            new_trajectory_memory,
            new_trajectory_valence,
            new_trajectory_reuse,
            new_previous_surface,
        ) = new_v71_state

        continuity = torch.nn.functional.cosine_similarity(
            torch.tanh(new_ancestry_state),
            torch.tanh(new_active_state),
            dim=-1,
        ).view(-1, 1)
        projection_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(new_active_state),
            torch.tanh(new_projection_state),
            dim=-1,
        ).view(-1, 1)
        transition_reuse = torch.nn.functional.cosine_similarity(
            torch.tanh(new_trajectory_memory),
            torch.tanh(new_trajectory_reuse),
            dim=-1,
        ).view(-1, 1)
        transformed_signal = transform * torch.relu(projection_alignment)
        rss_norm = torch.norm(new_previous_surface, dim=-1, keepdim=True) / math.sqrt(max(self.hidden_size, 1))
        rss_gain = torch.clamp((rss_norm - self.rss_floor) / max(self.rss_target - self.rss_floor, 1e-6), 0.0, 1.0)
        carrier_pressure = torch.clamp(
            torch.norm(new_long_carrier, dim=-1, keepdim=True) / math.sqrt(max(self.carrier_dim, 1)) / max(self.trajectory_carrier_norm_cap, 1e-6),
            0.0,
            2.0,
        )
        stasis_cost = torch.relu(self.rss_floor - rss_norm)
        permeability_cost = torch.relu(permeability - 0.62)
        resource_earned = (
            0.30 * torch.relu(continuity)
            + 0.25 * torch.relu(projection_alignment)
            + 0.20 * torch.relu(transition_reuse)
            + 0.15 * transformed_signal
            + 0.10 * rss_gain
        )
        resource_spent = (
            0.35 * anti_self_pressure * internalize
            + 0.25 * carrier_pressure
            + 0.20 * stasis_cost
            + 0.20 * permeability_cost
        )
        resource_delta = effective_resource_gain * resource_earned - self.resource_cost_scale * resource_spent
        new_resource_state = torch.clamp(
            self.resource_retention * resource_state + resource_delta,
            min=-4.0,
            max=4.0,
        )
        new_resource_open = torch.sigmoid(new_resource_state)
        death_pressure = torch.clamp(
            0.45 * stasis_cost + 0.35 * anti_self_pressure * internalize + 0.20 * torch.relu(0.20 - new_resource_open),
            0.0,
            1.0,
        )
        survival_pressure = torch.clamp(resource_earned - resource_spent, -1.0, 1.0)

        self._step_aux.update(
            {
                "v72_resource_state_mean": float(new_resource_state.mean().item()),
                "v72_resource_open_mean": float(new_resource_open.mean().item()),
                "v72_resource_earned_mean": float(resource_earned.mean().item()),
                "v72_resource_spent_mean": float(resource_spent.mean().item()),
                "v72_survival_pressure_mean": float(survival_pressure.mean().item()),
                "v72_death_pressure_mean": float(death_pressure.mean().item()),
                "v72_anti_self_pressure_mean": float(anti_self_pressure.mean().item()),
                "v72_environment_constraint_mean": float(constraint.mean().item()),
                "v72_rss_norm_mean": float(rss_norm.mean().item()),
                "v72_rss_gain_mean": float(rss_gain.mean().item()),
                "v72_carrier_pressure_mean": float(carrier_pressure.mean().item()),
                "v72_surface_discontinuity_mean": float(surface_discontinuity.mean().item()),
                "v72_resource_authority_mean": float(resource_authority.mean().item()),
            }
        )
        return (*new_v71_state, new_resource_state)


class DemianNativeV72SurfaceAblationSubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: resource is internal but absent from exposed surface."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, v72_exposure_scale=0.0, **kwargs)


class DemianNativeV72LowExposureSubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: resource has reduced exposed-surface tint."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, v72_exposure_scale=0.01, **kwargs)


class DemianNativeV72RefractorySubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: surface shocks temporarily reduce resource authority."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            refractory_surface_delta_threshold=0.14,
            refractory_resource_gain_scale=0.15,
            **kwargs,
        )


class DemianNativeV72ObserverOnlySubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: resource is measured but has no state authority."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            resource_gain=0.0,
            environment_constraint_scale=0.0,
            anti_self_scale=0.0,
            resource_tightness_scale=0.0,
            v72_exposure_scale=0.0,
            **kwargs,
        )


class DemianNativeV72TightnessOnlySubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: isolate resource-open pressure into tightness."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            resource_gain=0.0,
            environment_constraint_scale=0.0,
            anti_self_scale=0.0,
            v72_exposure_scale=0.0,
            **kwargs,
        )


class DemianNativeV72ConstraintOnlySubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: keep scarcity constraint without resource drive."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            resource_gain=0.0,
            anti_self_scale=0.0,
            v72_exposure_scale=0.0,
            **kwargs,
        )


class DemianNativeV72DriveOnlySubstrate(DemianNativeV72Substrate):
    """Native v7.2 probe: keep direct resource drive without scarcity constraint."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(
            hidden_size,
            environment_constraint_scale=0.0,
            anti_self_scale=0.0,
            v72_exposure_scale=0.0,
            **kwargs,
        )


class DemianNativeV74Substrate(DemianNativeV72Substrate):
    """Native v7.4: ownership/viability tension transformed by self-policy."""

    def __init__(
        self,
        hidden_size: int,
        self_potential_retention: float = 0.88,
        quarantine_retention: float = 0.82,
        self_policy_scale: float = 0.06,
        ancestry_write_scale: float = 0.018,
        recovery_drive_scale: float = 0.045,
        refusal_boundary_scale: float = 0.035,
        quarantine_scale: float = 0.055,
        self_topology_scale: float = 0.010,
        hold_topology_scale: float = 0.008,
        dynamic_topology_coupling_scale: float = 0.096,
        dynamic_step_min: float = 0.96,
        dynamic_step_max: float = 1.01,
        dynamic_step_hold_slowdown: float = 0.03,
        dynamic_step_refusal_slowdown: float = 0.02,
        dynamic_step_recovery_accel: float = 0.015,
        dynamic_step_integrate_accel: float = 0.010,
        pressure_policy_blend: float = 0.0,
        pressure_policy_floor: float = 0.01,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.self_potential_retention = self_potential_retention
        self.quarantine_retention = quarantine_retention
        self.self_policy_scale = self_policy_scale
        self.ancestry_write_scale = ancestry_write_scale
        self.recovery_drive_scale = recovery_drive_scale
        self.refusal_boundary_scale = refusal_boundary_scale
        self.quarantine_scale = quarantine_scale
        self.self_topology_scale = self_topology_scale
        self.hold_topology_scale = hold_topology_scale
        self.dynamic_topology_coupling_scale = dynamic_topology_coupling_scale
        self.dynamic_step_min = dynamic_step_min
        self.dynamic_step_max = dynamic_step_max
        self.dynamic_step_hold_slowdown = dynamic_step_hold_slowdown
        self.dynamic_step_refusal_slowdown = dynamic_step_refusal_slowdown
        self.dynamic_step_recovery_accel = dynamic_step_recovery_accel
        self.dynamic_step_integrate_accel = dynamic_step_integrate_accel
        self.pressure_policy_blend = pressure_policy_blend
        self.pressure_policy_floor = pressure_policy_floor

        feature_dim = hidden_size * 9 + 8
        self.self_potential_mix = nn.Linear(feature_dim, hidden_size)
        self.self_policy_head = nn.Linear(feature_dim + hidden_size * 2 + 1, 7)
        self.policy_to_active = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.policy_to_boundary = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.policy_to_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.policy_to_ancestry = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        rng_state = torch.random.get_rng_state()
        self.policy_to_topology_shadow = nn.Linear(hidden_size * 2, self.topology_edges, bias=False)
        torch.random.set_rng_state(rng_state)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        base = super().initial_state(batch_size, device)
        self_potential = torch.zeros(batch_size, self.hidden_size, device=device)
        quarantine_state = torch.zeros(batch_size, self.hidden_size, device=device)
        topology_shadow = torch.zeros(batch_size, self.topology_edges, device=device)
        return (*base, self_potential, quarantine_state, topology_shadow)

    def _split_v74_state(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(state) == 22:
            topology_shadow = torch.zeros(
                state[20].shape[0],
                self.topology_edges,
                device=state[20].device,
                dtype=state[20].dtype,
            )
            return state[:20], state[20], state[21], topology_shadow
        return state[:20], state[20], state[21], state[22]

    def inject_coupling_message(self, state, message, strength):
        if not hasattr(self, 'coupling_to_potential'):
            h = self.hidden_size
            dev = message.device
            self.register_buffer('coupling_to_potential',
                torch.randn(h, h, device=dev) * 0.1)
            self.register_buffer('coupling_to_topology',
                torch.randn(h, self.topology_edges, device=dev) * 0.1)
            self.register_buffer('coupling_to_quarantine',
                torch.randn(h, h, device=dev) * 0.1)
        v72_state, self_potential, quarantine_state, topology_shadow = self._split_v74_state(state)
        base = super().inject_coupling_message(v72_state, message, strength * 0.5)
        msg_flat = message.view(-1)
        pot_delta = (msg_flat @ self.coupling_to_potential).view(1, -1)
        q_delta = (msg_flat @ self.coupling_to_quarantine).view(1, -1)
        topo_delta = (msg_flat @ self.coupling_to_topology).view(1, -1)
        new_potential = self_potential + strength * torch.tanh(pot_delta)
        new_quarantine = quarantine_state + strength * torch.tanh(q_delta)
        new_topology = topology_shadow + strength * torch.tanh(topo_delta)
        if isinstance(base, tuple):
            return (*base, new_potential, new_quarantine, new_topology)
        return (base, new_potential, new_quarantine, new_topology)

    def state_components(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        if len(state) <= 20:
            return super().state_components(state)
        v72_state, self_potential, quarantine_state, topology_shadow = self._split_v74_state(state)
        components = super().state_components(v72_state)
        components["self_potential"] = self_potential
        components["quarantine"] = quarantine_state
        components["dynamic_topology"] = topology_shadow
        return components

    def state_vector(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(state) <= 20:
            return super().state_vector(state)
        v72_state, self_potential, quarantine_state, topology_shadow = self._split_v74_state(state)
        base_view = super().state_vector(v72_state)
        return base_view + 0.015 * torch.tanh(self_potential - quarantine_state)

    def step(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        v72_state, self_potential, quarantine_state, topology_shadow = self._split_v74_state(state)
        v71_state, resource_state = self._split_v72_state(v72_state)
        (
            fast,
            slow,
            long_carrier,
            short_support,
            packet,
            control_short,
            control_long,
            tightness,
            controller_state,
            conflict_potential,
            topology_state,
            ancestry_state,
            active_state,
            projection_state,
            boundary_state,
            trajectory_memory,
            trajectory_valence,
            trajectory_reuse,
            previous_surface,
        ) = v71_state

        surface = DemianNativeV71Substrate.state_vector(self, v71_state)
        ownership_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(active_state + trajectory_memory + trajectory_reuse),
            torch.tanh(ancestry_state + trajectory_valence),
            dim=-1,
        ).view(-1, 1)
        ownership = torch.clamp(0.5 + 0.5 * ownership_alignment, 0.0, 1.0)
        resource_open = torch.sigmoid(resource_state)
        rss_norm = torch.norm(previous_surface, dim=-1, keepdim=True) / math.sqrt(max(self.hidden_size, 1))
        rss_gain = torch.clamp((rss_norm - self.rss_floor) / max(self.rss_target - self.rss_floor, 1e-6), 0.0, 1.0)
        projection_alignment = torch.nn.functional.cosine_similarity(
            torch.tanh(active_state),
            torch.tanh(projection_state),
            dim=-1,
        ).view(-1, 1)
        viability = torch.clamp(
            0.40 * resource_open
            + 0.25 * rss_gain
            + 0.20 * torch.relu(projection_alignment)
            + 0.15 * torch.relu(torch.nn.functional.cosine_similarity(
                torch.tanh(trajectory_memory),
                torch.tanh(trajectory_reuse),
                dim=-1,
            ).view(-1, 1)),
            0.0,
            1.0,
        )

        tension = torch.abs(viability - ownership)
        p_integrate = viability * ownership
        p_refuse = viability * (1.0 - ownership)
        p_recover = (1.0 - viability) * ownership
        p_reject = (1.0 - viability) * (1.0 - ownership)
        pressure_stack = torch.cat([p_integrate, p_refuse, p_recover, p_reject], dim=-1)
        pressure_distribution = pressure_stack / (pressure_stack.sum(dim=-1, keepdim=True) + 1e-8)
        pressure_entropy = -torch.sum(
            pressure_distribution * torch.log(pressure_distribution + 1e-8),
            dim=-1,
            keepdim=True,
        ) / math.log(4.0)
        hold_pressure = tension * pressure_entropy
        pressure_scalars = torch.cat(
            [
                viability,
                ownership,
                tension,
                p_integrate,
                p_refuse,
                p_recover,
                p_reject,
                resource_open,
            ],
            dim=-1,
        )
        policy_features = torch.cat(
            [
                torch.tanh(surface),
                torch.tanh(ancestry_state),
                torch.tanh(active_state),
                torch.tanh(projection_state),
                torch.tanh(boundary_state),
                torch.tanh(trajectory_memory),
                torch.tanh(trajectory_valence),
                torch.tanh(trajectory_reuse),
                torch.tanh(quarantine_state),
                pressure_scalars,
            ],
            dim=-1,
        )
        potential_candidate = torch.tanh(self.self_potential_mix(policy_features))
        provisional_self_potential = (
            self.self_potential_retention * self_potential
            + (1.0 - self.self_potential_retention) * tension * potential_candidate
        )
        policy_logits = self.self_policy_head(
            torch.cat(
                [
                    policy_features,
                    torch.tanh(provisional_self_potential),
                    torch.tanh(self_potential - quarantine_state),
                    hold_pressure,
                ],
                dim=-1,
            )
        )
        policy_temperature = 1.0 + 2.0 * tension - 0.75 * hold_pressure
        learned_policy = torch.softmax(policy_logits * policy_temperature, dim=-1)
        pressure_policy_raw = torch.cat(
            [
                (1.0 - tension) * viability * ownership,  # preserve
                tension * (0.5 + 0.5 * viability),  # adapt
                p_recover,  # recover
                p_refuse,  # refuse
                p_reject + 0.50 * p_refuse,  # quarantine
                p_integrate,  # integrate
                hold_pressure,  # hold
            ],
            dim=-1,
        )
        pressure_policy_raw = torch.clamp(
            pressure_policy_raw + self.pressure_policy_floor,
            min=1e-8,
        )
        pressure_policy = pressure_policy_raw / pressure_policy_raw.sum(dim=-1, keepdim=True)
        pressure_policy_blend = torch.clamp(
            torch.as_tensor(
                self.pressure_policy_blend,
                device=learned_policy.device,
                dtype=learned_policy.dtype,
            ),
            min=0.0,
            max=1.0,
        )
        policy = (1.0 - pressure_policy_blend) * learned_policy + pressure_policy_blend * pressure_policy
        preserve_gate = policy[:, 0:1]
        adapt_gate = policy[:, 1:2]
        recover_gate = policy[:, 2:3]
        refuse_gate = policy[:, 3:4]
        quarantine_gate = policy[:, 4:5]
        integrate_gate = policy[:, 5:6]
        hold_gate = policy[:, 6:7]

        resolution_open = 1.0 - hold_gate * torch.clamp(0.35 + pressure_entropy, 0.0, 1.0)
        step_dt = torch.clamp(
            1.0
            + self.dynamic_step_recovery_accel * recover_gate * p_recover
            + self.dynamic_step_integrate_accel * integrate_gate * p_integrate
            - self.dynamic_step_hold_slowdown * hold_pressure
            - self.dynamic_step_refusal_slowdown * refuse_gate * p_refuse
            - 0.20 * hold_gate * pressure_entropy,
            min=self.dynamic_step_min,
            max=self.dynamic_step_max,
        )
        new_self_potential = self_potential + step_dt * (provisional_self_potential - self_potential)
        owned_write = resolution_open * integrate_gate * p_integrate
        recovery_write = resolution_open * recover_gate * p_recover
        refusal_write = resolution_open * refuse_gate * p_refuse
        quarantine_write = quarantine_gate * (p_refuse + p_reject) + hold_gate * hold_pressure
        policy_drive = torch.tanh(torch.cat([new_self_potential, quarantine_state], dim=-1))
        active_policy = torch.tanh(self.policy_to_active(policy_drive))
        boundary_policy = torch.tanh(self.policy_to_boundary(policy_drive))
        projection_policy = torch.tanh(self.policy_to_projection(policy_drive))
        ancestry_policy = torch.tanh(self.policy_to_ancestry(policy_drive))
        topology_policy = torch.tanh(self.policy_to_topology_shadow(policy_drive))
        transformed_projection = torch.tanh(projection_state + trajectory_memory)
        topology_authority = (
            resolution_open
            * ownership
            * (1.0 - hold_gate)
            * torch.clamp(p_integrate + p_recover + 0.25 * adapt_gate * tension, 0.0, 1.0)
        )
        dynamic_topology_injection = (
            self.dynamic_topology_coupling_scale
            * step_dt
            * topology_authority
            * torch.tanh(topology_shadow + 0.35 * topology_policy)
        )

        adjusted_v71_state = (
            fast
            + step_dt * self.self_policy_scale * (
                adapt_gate * active_policy
                + recovery_write * torch.tanh(ancestry_state - active_state)
                - refusal_write * torch.tanh(projection_state)
                + 0.25 * hold_gate * torch.tanh(self_potential)
            ),
            slow,
            long_carrier,
            short_support,
            packet * (1.0 - 0.04 * step_dt * refusal_write),
            control_short,
            control_long,
            tightness + step_dt * (0.025 * recovery_write - 0.012 * owned_write),
            controller_state,
            conflict_potential + step_dt * 0.05 * tension * torch.tanh(conflict_potential + 1.0),
            torch.clamp(topology_state + dynamic_topology_injection, min=-1.0, max=1.0),
            ancestry_state + step_dt * self.ancestry_write_scale * owned_write * ancestry_policy,
            active_state
            + step_dt * (
                self.recovery_drive_scale * recovery_write * torch.tanh(ancestry_state - active_state)
                + self.self_policy_scale * resolution_open * adapt_gate * active_policy
                + 0.5 * self.self_policy_scale * owned_write * torch.tanh(transformed_projection - active_state)
            ),
            projection_state
            + step_dt * (
                self.self_policy_scale * adapt_gate * projection_policy
                - self.refusal_boundary_scale * refusal_write * torch.tanh(projection_state - quarantine_state)
            ),
            boundary_state
            + step_dt * (
                self.refusal_boundary_scale * refusal_write * boundary_policy
                + 0.5 * self.refusal_boundary_scale * quarantine_write * torch.tanh(boundary_state)
            ),
            trajectory_memory,
            trajectory_valence,
            trajectory_reuse,
            previous_surface,
        )
        adjusted_v72_state = (*adjusted_v71_state, resource_state)
        stepped_v72_state = super().step(adjusted_v72_state)
        new_v72_state = tuple(
            adjusted + step_dt * (stepped - adjusted)
            for adjusted, stepped in zip(adjusted_v72_state, stepped_v72_state)
        )
        new_v71_state, _new_resource_state = self._split_v72_state(new_v72_state)
        new_trajectory_memory = new_v71_state[15]
        topology_pressure = (
            self.self_topology_scale * quarantine_write
            + self.hold_topology_scale * hold_pressure
        )
        topology_memory_pressure = torch.clamp(
            torch.nn.functional.cosine_similarity(
                torch.tanh(new_trajectory_memory),
                torch.tanh(new_self_potential),
                dim=-1,
            ).view(-1, 1),
            min=0.0,
            max=1.0,
        )
        v74_topology_drive = step_dt * (topology_pressure * topology_policy + (
            0.25 * self.self_topology_scale * topology_memory_pressure * torch.tanh(
                self.trajectory_to_topology(torch.tanh(new_self_potential))
            )
        ))
        new_topology_shadow = torch.clamp(
            torch.pow(torch.as_tensor(0.94, device=topology_shadow.device, dtype=topology_shadow.dtype), step_dt)
            * topology_shadow + v74_topology_drive,
            min=-1.0,
            max=1.0,
        )
        new_quarantine_state = (
            torch.pow(
                torch.as_tensor(self.quarantine_retention, device=quarantine_state.device, dtype=quarantine_state.dtype),
                step_dt,
            )
            * quarantine_state
            + step_dt * (
                self.quarantine_scale * quarantine_write * torch.tanh(projection_state - active_state)
                - self.quarantine_scale * owned_write * torch.tanh(quarantine_state)
            )
        )

        self._step_aux.update(
            {
                "v74_viability_mean": float(viability.mean().item()),
                "v74_ownership_mean": float(ownership.mean().item()),
                "v74_tension_mean": float(tension.mean().item()),
                "v74_self_potential_norm": float(torch.norm(new_self_potential).item()),
                "v74_quarantine_norm": float(torch.norm(new_quarantine_state).item()),
                "v74_integrate_pressure_mean": float(p_integrate.mean().item()),
                "v74_refuse_pressure_mean": float(p_refuse.mean().item()),
                "v74_recover_pressure_mean": float(p_recover.mean().item()),
                "v74_reject_pressure_mean": float(p_reject.mean().item()),
                "v74_hold_pressure_mean": float(hold_pressure.mean().item()),
                "v74_pressure_entropy_mean": float(pressure_entropy.mean().item()),
                "v74_preserve_gate_mean": float(preserve_gate.mean().item()),
                "v74_adapt_gate_mean": float(adapt_gate.mean().item()),
                "v74_recover_gate_mean": float(recover_gate.mean().item()),
                "v74_refuse_gate_mean": float(refuse_gate.mean().item()),
                "v74_quarantine_gate_mean": float(quarantine_gate.mean().item()),
                "v74_integrate_gate_mean": float(integrate_gate.mean().item()),
                "v74_hold_gate_mean": float(hold_gate.mean().item()),
                "v74_resolution_open_mean": float(resolution_open.mean().item()),
                "v74_dynamic_step_dt_mean": float(step_dt.mean().item()),
                "v74_dynamic_step_dt_min": float(step_dt.min().item()),
                "v74_dynamic_step_dt_max": float(step_dt.max().item()),
                "v74_ancestry_write_mean": float(owned_write.mean().item()),
                "v74_topology_drive_norm": float(torch.norm(v74_topology_drive).item()),
                "v74_topology_pressure_mean": float(topology_pressure.mean().item()),
                "v74_topology_memory_pressure_mean": float(topology_memory_pressure.mean().item()),
                "v74_topology_state_norm": float(torch.norm(new_topology_shadow).item()),
                "v74_dynamic_topology_authority_mean": float(topology_authority.mean().item()),
                "v74_dynamic_topology_injection_norm": float(torch.norm(dynamic_topology_injection).item()),
            }
        )
        return (*new_v72_state, new_self_potential, new_quarantine_state, new_topology_shadow)


class DemianNativeV74PressurePolicySubstrate(DemianNativeV74Substrate):
    """Native v7.4 probe: replace learned self-policy gates with pressure gates."""

    def __init__(self, hidden_size: int, **kwargs):
        kwargs.setdefault("pressure_policy_blend", 1.0)
        super().__init__(hidden_size, **kwargs)


class DemianNativeV74CalibratedPressurePolicySubstrate(DemianNativeV74Substrate):
    """Native v7.4 probe: partial pressure routing at the current calibration elbow."""

    def __init__(self, hidden_size: int, **kwargs):
        kwargs.setdefault("pressure_policy_blend", 0.2)
        kwargs.setdefault("pressure_policy_floor", 0.005)
        super().__init__(hidden_size, **kwargs)


class DemianNativeV8Substrate(DemianNativeV2Substrate):
    """Native v8: minimal genotype substrate.

    v2 channel anatomy + tightness constraint modulator.
    No organs above v3. No policy gates. No named selves.
    No trajectory memory. No topology state. No quarantine.

    Seven channels (fast, slow, long_carrier, short_support, packet,
    control_short, control_long) + tightness scalar.
    Identical recurrence applied each step.
    Environmental coupling through state_vector surface only.

    Hypothesis: a compressed recurrent encoding with a single endogenous
    constraint modulator produces richer attractor structure than the
    organ-heavy v7.4 architecture.
    """

    def __init__(
        self,
        hidden_size: int,
        tightness_retention: float = 0.90,
        initial_tightness_bias: float = -0.35,
        tightness_to_carrier_slow_scale: float = 0.22,
        tightness_to_packet_slow_scale: float = 0.18,
        tightness_to_control_slow_scale: float = 0.12,
        tightness_to_release_gain_scale: float = 0.08,
        tightness_to_release_threshold_scale: float = 0.08,
        tightness_to_support_packet_scale: float = 0.025,
        tightness_to_carrier_decay_scale: float = 0.018,
        tightness_to_control_packet_scale: float = 0.035,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.tightness_retention = tightness_retention
        self.initial_tightness_bias = initial_tightness_bias
        self.tightness_to_carrier_slow_scale = tightness_to_carrier_slow_scale
        self.tightness_to_packet_slow_scale = tightness_to_packet_slow_scale
        self.tightness_to_control_slow_scale = tightness_to_control_slow_scale
        self.tightness_to_release_gain_scale = tightness_to_release_gain_scale
        self.tightness_to_release_threshold_scale = tightness_to_release_threshold_scale
        self.tightness_to_support_packet_scale = tightness_to_support_packet_scale
        self.tightness_to_carrier_decay_scale = tightness_to_carrier_decay_scale
        self.tightness_to_control_packet_scale = tightness_to_control_packet_scale

        full_dim = (
            hidden_size
            + hidden_size
            + self.carrier_dim
            + self.support_dim
            + self.packet_dim
            + self.control_dim
            + self.control_dim
            + 1
        )
        self.tightness_mix = nn.Linear(full_dim, 1)
        self.tightness_readout = nn.Linear(1, hidden_size, bias=False)
        self.tightness_expose_gate = nn.Linear(hidden_size * 7, hidden_size)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long = super().initial_state(batch_size, device)
        tightness = torch.full((batch_size, 1), self.initial_tightness_bias, device=device)
        return fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        return {
            "fast": fast,
            "slow": slow,
            "carrier": long_carrier,
            "short_support": short_support,
            "packet": packet,
            "control_short": control_short,
            "control_long": control_long,
            "tightness": tightness,
            "message": long_carrier,
        }

    def inject_coupling_message(self, state, message, strength):
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        base_fast = fast + strength * 0.5 * message.view(1, -1)
        new_tightness = tightness + strength * 0.3 * torch.tanh(message.sum(dim=-1, keepdim=True))
        return (base_fast, slow, long_carrier, short_support, packet, control_short, control_long, new_tightness)

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        slow_view = torch.tanh(self.slow_readout(slow))
        carrier_view = torch.tanh(self.long_carrier_readout(long_carrier))
        support_view = torch.tanh(self.short_support_readout(short_support))
        packet_view = torch.tanh(self.packet_readout(packet))
        cs_view = self.control_read_scale * torch.tanh(self.control_short_readout(control_short))
        cl_view = self.control_read_scale * torch.tanh(self.control_long_readout(control_long))
        tight_view = torch.tanh(self.tightness_readout(torch.tanh(tightness)))
        gate = torch.sigmoid(
            self.tightness_expose_gate(
                torch.cat([fast, slow_view, carrier_view, support_view, packet_view, cs_view + cl_view, tight_view], dim=-1)
            )
        )
        anchored = (
            0.22 * slow_view
            + 0.25 * carrier_view
            + 0.12 * support_view
            + 0.10 * packet_view
            + 0.10 * cs_view
            + 0.09 * cl_view
            + 0.12 * tight_view
        )
        return gate * fast + (1.0 - gate) * anchored

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, short_support, packet, control_short, control_long], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(stacked))
        slow_gate = torch.sigmoid(self.slow_gate(stacked))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(stacked))
        support_gate = torch.sigmoid(self.short_support_gate(stacked))
        packet_gate = torch.sigmoid(self.packet_gate(stacked))
        control_short_gate = torch.sigmoid(self.control_short_gate(stacked))
        control_long_gate = torch.sigmoid(self.control_long_gate(stacked))

        fast_candidate = torch.tanh(self.fast_mix(stacked))
        slow_candidate = torch.tanh(self.slow_mix(stacked))
        carrier_candidate = torch.tanh(self.long_carrier_mix(stacked))
        support_candidate = torch.tanh(self.short_support_mix(stacked))
        packet_candidate = torch.tanh(self.packet_mix(stacked))
        control_short_candidate = torch.tanh(self.control_short_mix(stacked))
        control_long_candidate = torch.tanh(self.control_long_mix(stacked))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([stacked, tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        control_bridge = torch.tanh(0.85 * control_short + 0.75 * control_long)
        long_control_bridge = torch.tanh(control_long)
        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_support_to_packet_scale = torch.clamp(
            self.support_to_packet_scale + self.tightness_to_support_packet_scale * loose_open,
            min=0.0,
        )
        eff_control_to_slow_scale = self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open
        eff_packet_to_slow_scale = self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open
        eff_carrier_to_slow_scale = self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open
        eff_release_gain = torch.clamp(
            self.release_gain - self.tightness_to_release_gain_scale * tightness_open,
            min=0.0,
        )
        eff_release_threshold = torch.clamp(
            self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open,
            min=0.0,
            max=1.0,
        )
        eff_long_carrier_decay = torch.clamp(
            self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open,
            max=0.999,
        )

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_bridge[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_bridge[:, : self.carrier_dim])
        control_long_to_carrier = self.control_long_to_carrier_scale * torch.tanh(long_control_bridge[:, : self.carrier_dim])
        support_to_packet = eff_support_to_packet_scale * torch.tanh(short_support[:, : self.packet_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(
            slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout + self.control_long_readout(control_long)
        )

        release_open = torch.sigmoid(self.release_gate(stacked))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_control_long = self.control_long_decay * control_long + control_long_gate * control_long_candidate
        new_short_support = (
            self.short_support_decay * short_support
            + support_gate * support_candidate
            + self.support_feedback_scale * torch.tanh(control_short[:, : self.support_dim])
        )
        new_packet = (
            self.packet_decay * packet
            + packet_gate * packet_candidate
            + support_to_packet
            + control_to_packet
            + endogenous_release
        )
        new_long_carrier = (
            eff_long_carrier_decay * long_carrier
            + carrier_gate * carrier_candidate
            + packet_to_carrier
            + control_to_carrier
            + control_long_to_carrier
        )
        new_slow = (
            self.slow_decay * slow
            + slow_gate * slow_candidate
            + packet_to_slow
            + control_to_slow
            + carrier_to_slow
            + 0.05 * torch.tanh(self.slow_readout(slow))
        )

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.support_to_fast_scale * torch.tanh(self.short_support_readout(new_short_support))
            + self.control_to_fast_scale * torch.tanh(
                self.control_short_readout(new_control_short) + self.control_long_readout(new_control_long)
            )
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "carrier_write_mean": float(carrier_gate.mean().item()),
            "short_support_write_mean": float(support_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "control_long_write_mean": float(control_long_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier + control_long_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "carrier_short_residual_norm": float(torch.norm(self.short_support_decay * short_support).item()),
            "carrier_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
            "effective_control_to_packet_scale_mean": float(eff_control_to_packet_scale.mean().item()),
            "effective_support_to_packet_scale_mean": float(eff_support_to_packet_scale.mean().item()),
            "effective_control_to_slow_scale_mean": float(eff_control_to_slow_scale.mean().item()),
            "effective_packet_to_slow_scale_mean": float(eff_packet_to_slow_scale.mean().item()),
            "effective_carrier_to_slow_scale_mean": float(eff_carrier_to_slow_scale.mean().item()),
            "effective_long_carrier_decay_mean": float(eff_long_carrier_decay.mean().item()),
            "effective_release_gain_mean": float(eff_release_gain.mean().item()),
            "effective_release_threshold_mean": float(eff_release_threshold.mean().item()),
        }
        return new_fast, new_slow, new_long_carrier, new_short_support, new_packet, new_control_short, new_control_long, new_tightness


class DemianNativeV85Substrate(DemianNativeV8Substrate):
    """Native v8.5: minimal 5-channel genotype.

    Drops short_support and control_long. Keeps:
    fast, slow, long_carrier, packet, control_short, tightness.

    Minimum viable channel set per ablation results.
    """

    def __init__(
        self,
        hidden_size: int,
        tightness_retention: float = 0.90,
        initial_tightness_bias: float = -0.35,
        tightness_to_carrier_slow_scale: float = 0.22,
        tightness_to_packet_slow_scale: float = 0.18,
        tightness_to_control_slow_scale: float = 0.12,
        tightness_to_release_gain_scale: float = 0.08,
        tightness_to_release_threshold_scale: float = 0.08,
        tightness_to_carrier_decay_scale: float = 0.018,
        tightness_to_control_packet_scale: float = 0.035,
        **kwargs,
    ):
        kwargs.setdefault("support_to_packet_scale", 0.0)
        kwargs.setdefault("support_to_fast_scale", 0.0)
        kwargs.setdefault("control_long_to_carrier_scale", 0.0)
        kwargs.setdefault("support_feedback_scale", 0.0)
        super().__init__(hidden_size, **kwargs)
        self.support_to_packet_scale = 0.0
        self.support_to_fast_scale = 0.0
        self.control_long_to_carrier_scale = 0.0
        self.support_feedback_scale = 0.0

    def step(
        self,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        fast, slow, long_carrier, short_support, packet, control_short, control_long, tightness = state
        stacked = torch.cat([fast, slow, long_carrier, packet, control_short], dim=-1)
        surface = self.state_vector(state)

        fast_gate = torch.sigmoid(self.fast_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        slow_gate = torch.sigmoid(self.slow_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        carrier_gate = torch.sigmoid(self.long_carrier_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        packet_gate = torch.sigmoid(self.packet_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        control_short_gate = torch.sigmoid(self.control_short_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))

        fast_candidate = torch.tanh(self.fast_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        slow_candidate = torch.tanh(self.slow_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        carrier_candidate = torch.tanh(self.long_carrier_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        packet_candidate = torch.tanh(self.packet_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        control_short_candidate = torch.tanh(self.control_short_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))

        tightness_drive = torch.tanh(self.tightness_mix(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long), tightness], dim=-1)))
        new_tightness = self.tightness_retention * tightness + (1.0 - self.tightness_retention) * tightness_drive
        tightness_open = torch.sigmoid(new_tightness)
        loose_open = 1.0 - tightness_open

        carrier_readout = torch.tanh(self.long_carrier_readout(long_carrier))
        packet_readout = torch.tanh(self.packet_readout(packet))

        eff_control_to_packet_scale = self.control_to_packet_scale + self.tightness_to_control_packet_scale * loose_open
        eff_control_to_slow_scale = self.control_to_slow_scale + self.tightness_to_control_slow_scale * tightness_open
        eff_packet_to_slow_scale = self.packet_to_slow_scale + self.tightness_to_packet_slow_scale * tightness_open
        eff_carrier_to_slow_scale = self.carrier_to_slow_scale + self.tightness_to_carrier_slow_scale * tightness_open
        eff_release_gain = torch.clamp(self.release_gain - self.tightness_to_release_gain_scale * tightness_open, min=0.0)
        eff_release_threshold = torch.clamp(self.release_threshold + self.tightness_to_release_threshold_scale * tightness_open, min=0.0, max=1.0)
        eff_long_carrier_decay = torch.clamp(self.long_carrier_decay + self.tightness_to_carrier_decay_scale * tightness_open, max=0.999)

        control_to_packet = eff_control_to_packet_scale * torch.tanh(control_short[:, : self.packet_dim])
        control_to_carrier = self.control_to_carrier_scale * torch.tanh(control_short[:, : self.carrier_dim])
        packet_to_carrier = self.packet_to_carrier_scale * torch.tanh(packet[:, : self.carrier_dim])
        carrier_to_slow = eff_carrier_to_slow_scale * carrier_readout
        packet_to_slow = eff_packet_to_slow_scale * torch.tanh(packet_readout + 0.85 * carrier_readout)
        control_to_slow = eff_control_to_slow_scale * torch.tanh(slow_candidate + 0.75 * carrier_readout + 0.5 * packet_readout)

        release_open = torch.sigmoid(self.release_gate(torch.cat([fast, slow, long_carrier, torch.zeros_like(short_support), packet, control_short, torch.zeros_like(control_long)], dim=-1)))
        release_strength = eff_release_gain * torch.relu(release_open - eff_release_threshold)
        endogenous_release = release_strength * torch.tanh(self.coupling_proj(surface))

        new_control_short = self.control_short_decay * control_short + control_short_gate * control_short_candidate
        new_packet = self.packet_decay * packet + packet_gate * packet_candidate + control_to_packet + endogenous_release
        new_long_carrier = eff_long_carrier_decay * long_carrier + carrier_gate * carrier_candidate + packet_to_carrier + control_to_carrier
        new_slow = self.slow_decay * slow + slow_gate * slow_candidate + packet_to_slow + control_to_slow + carrier_to_slow + 0.05 * torch.tanh(self.slow_readout(slow))

        fast_bias = (
            self.slow_to_fast_scale * torch.tanh(new_slow)
            + self.carrier_to_fast_scale * torch.tanh(self.long_carrier_readout(new_long_carrier))
            + self.control_to_fast_scale * torch.tanh(self.control_short_readout(new_control_short))
            + 0.08 * torch.tanh(self.tightness_readout(torch.tanh(new_tightness)))
        )
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + fast_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "message_write_mean": float(carrier_gate.mean().item()),
            "packet_write_mean": float(packet_gate.mean().item()),
            "control_short_write_mean": float(control_short_gate.mean().item()),
            "release_open_mean": float(release_open.mean().item()),
            "release_strength_mean": float(release_strength.mean().item()),
            "endogenous_release_norm": float(torch.norm(endogenous_release).item()),
            "control_to_packet_norm": float(torch.norm(control_to_packet).item()),
            "control_to_carrier_norm": float(torch.norm(control_to_carrier).item()),
            "control_to_slow_norm": float(torch.norm(control_to_slow).item()),
            "carrier_to_slow_norm": float(torch.norm(carrier_to_slow).item()),
            "packet_to_carrier_norm": float(torch.norm(packet_to_carrier).item()),
            "packet_to_slow_norm": float(torch.norm(packet_to_slow).item()),
            "carrier_long_residual_norm": float(torch.norm(eff_long_carrier_decay * long_carrier).item()),
            "tightness_mean": float(tightness_open.mean().item()),
            "tightness_state_mean": float(new_tightness.mean().item()),
        }
        return fast, new_slow, new_long_carrier, torch.zeros_like(short_support), new_packet, new_control_short, torch.zeros_like(control_long), new_tightness


class DemianNativeV9Substrate(SelfLoopSubstrate):
    """Native v9: minimal 3-channel genotype.

    Collapsed from 7 channels (v8) to exactly 3 with single dynamical roles:
      - fast:     exposed surface (state_vector returns this directly)
      - slow:     deep continuity (only input from fast projection)
      - control:  singular steering signal (projects back onto fast bias)

    No carrier, packet, support, control_long, tightness, endogenous release.
    Each gate proiects from its own channel only — cross-coupling is through
    explicit additive bias terms.

    Hypothesis: a 3-channel self-loop with strict directional coupling
    produces richer attractor structure than 7-channel redundancy.
    Test: does 3-channel collapse to fixed point, or maintain dynamics?
    """

    def __init__(
        self,
        hidden_size: int,
        control_dim: int | None = None,
        slow_decay: float = 0.92,
        control_decay: float = 0.68,
        slow_readout_scale: float = 0.28,
        control_to_fast_scale: float = 0.12,
        fast_to_slow_gate_bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(hidden_size, **kwargs)
        self.control_dim = control_dim or max(4, hidden_size // 8)
        self.slow_decay = slow_decay
        self.control_decay = control_decay
        self.slow_readout_scale = slow_readout_scale
        self.control_to_fast_scale = control_to_fast_scale
        self.fast_to_slow_gate_bias = fast_to_slow_gate_bias

        # Fast gate and candidate — projects from fast only
        self.fast_gate = nn.Linear(hidden_size, hidden_size)
        self.fast_mix = nn.Linear(hidden_size, hidden_size)

        # Slow gate and candidate — projects from slow only
        self.slow_gate = nn.Linear(hidden_size, hidden_size)
        self.slow_mix = nn.Linear(hidden_size, hidden_size)

        # Control gate and candidate — projects from control only
        self.control_gate = nn.Linear(self.control_dim, self.control_dim)
        self.control_mix = nn.Linear(self.control_dim, self.control_dim)

        # Cross-coupling projection
        self.fast_to_slow = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fast_slow_to_control = nn.Linear(hidden_size + hidden_size, self.control_dim)
        self.control_readout = nn.Linear(self.control_dim, hidden_size, bias=False)

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        slow = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        control = torch.randn(batch_size, self.control_dim, device=device) * self.init_scale
        return fast, slow, control

    def state_components(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        fast, slow, control = state
        return {"fast": fast, "slow": slow, "control": control}

    def state_vector(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return state[0]

    def inject_coupling_message(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        message: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, control = state
        delta = strength * message
        if delta.shape != fast.shape:
            delta = delta.view(fast.shape)
        return fast + delta, slow, control

    def step(
        self,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fast, slow, control = state

        # Slow update — receives bias from fast readout
        slow_gate = torch.sigmoid(self.slow_gate(slow))
        slow_candidate = torch.tanh(self.slow_mix(slow))
        fast_to_slow_bias = self.slow_readout_scale * torch.tanh(self.fast_to_slow(fast))
        new_slow = self.slow_decay * slow + slow_gate * (slow_candidate + fast_to_slow_bias)

        # Control update — receives bias from fast+slow concatenated
        control_gate = torch.sigmoid(self.control_gate(control))
        control_candidate = torch.tanh(self.control_mix(control))
        fast_slow_bias = torch.tanh(self.fast_slow_to_control(torch.cat([fast, slow], dim=-1)))
        new_control = self.control_decay * control + control_gate * (control_candidate + fast_slow_bias)

        # Fast update — receives control bias (gated overwrite, like v8 fast)
        fast_gate = torch.sigmoid(self.fast_gate(fast))
        fast_candidate = torch.tanh(self.fast_mix(fast))
        control_bias = self.control_to_fast_scale * torch.tanh(self.control_readout(new_control))
        new_fast = (1.0 - fast_gate) * fast + fast_gate * self.state_gain * torch.tanh(fast_candidate + control_bias)

        self._step_aux = {
            "fast_update_mean": float(fast_gate.mean().item()),
            "slow_write_mean": float(slow_gate.mean().item()),
            "control_write_mean": float(control_gate.mean().item()),
            "fast_to_slow_bias_norm": float(torch.norm(fast_to_slow_bias).item()),
            "fast_slow_bias_norm": float(torch.norm(fast_slow_bias).item()),
            "control_bias_norm": float(torch.norm(control_bias).item()),
        }

        return new_fast, new_slow, new_control


class LSTMSubstrate(SelfLoopSubstrate):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        self.in_proj = nn.Linear(hidden_size, hidden_size)
        self.cell = nn.LSTMCell(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        c = torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale
        return h, c

    def state_vector(self, state: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h, c = state
        return 0.5 * (h + c)

    def step(self, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        x = self.feedback_scale * self.in_proj(h)
        nh, nc = self.cell(x, (h, c))
        return self.state_gain * nh, self.state_gain * nc


class DiagonalSSMSubstrate(SelfLoopSubstrate):
    """Small diagonal SSM with fixed decay structure and learnable couplings."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        base = torch.linspace(-2.5, -0.05, hidden_size)
        self.A_log = nn.Parameter(base)
        self.in_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mix = nn.Linear(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale

    def state_vector(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feedback_scale * self.mix(state)
        delta = torch.sigmoid(self.feedback_scale * self.out_proj(state)) * 0.15 + 0.01
        A = -torch.exp(self.A_log).unsqueeze(0)
        A_bar = torch.exp(delta * A)
        Bx = self.in_proj(x)
        return self.state_gain * (A_bar * state + (1.0 - A_bar) * torch.tanh(Bx))


class SelectiveSSMSubstrate(SelfLoopSubstrate):
    """Mamba-like selective SSM without the full language-model machinery."""

    def __init__(self, hidden_size: int, **kwargs):
        super().__init__(hidden_size, **kwargs)
        base = torch.linspace(-2.0, -0.03, hidden_size)
        self.A_log = nn.Parameter(base)
        self.input_gate = nn.Linear(hidden_size, hidden_size)
        self.forget_gate = nn.Linear(hidden_size, hidden_size)
        self.candidate = nn.Linear(hidden_size, hidden_size)
        self.read_gate = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.hidden_size, device=device) * self.init_scale

    def state_vector(self, state: torch.Tensor) -> torch.Tensor:
        return state

    def step(self, state: torch.Tensor) -> torch.Tensor:
        x = self.feedback_scale * state
        write = torch.sigmoid(self.input_gate(x))
        forget = torch.sigmoid(self.forget_gate(x))
        cand = torch.tanh(self.candidate(x))

        A = -torch.exp(self.A_log).unsqueeze(0)
        delta = 0.01 + 0.19 * forget
        A_bar = torch.exp(delta * A)

        next_state = A_bar * state + write * (1.0 - A_bar) * cand
        exposed = torch.sigmoid(self.read_gate(next_state)) * next_state
        return self.state_gain * (self.out_proj(exposed) + 0.25 * next_state)


SUBSTRATE_REGISTRY = {
    "rnn": VanillaRNNSubstrate,
    "gru": GRUSubstrate,
    "dual_gru": DualGRUSubstrate,
    "dual_gru_v2": DualGRUV2Substrate,
    "dual_gru_v3": DualGRUV3Substrate,
    "dual_gru_v3b": DualGRUV3BSubstrate,
    "dual_gru_v4": DualGRUV4Substrate,
    "dual_gru_v4m": DualGRUV4MemorySubstrate,
    "dual_gru_v5": DualGRUV5Substrate,
    "demian_native_v0": DemianNativeV0Substrate,
    "demian_native_v1": DemianNativeV1Substrate,
    "demian_native_v2": DemianNativeV2Substrate,
    "demian_native_v3": DemianNativeV3Substrate,
    "demian_native_v4": DemianNativeV4Substrate,
    "demian_native_v5": DemianNativeV5Substrate,
    "demian_native_v5.1": DemianNativeV51Substrate,
    "demian_native_v51": DemianNativeV51Substrate,
    "demian_native_v5.2": DemianNativeV52Substrate,
    "demian_native_v52": DemianNativeV52Substrate,
    "demian_native_v5.2b": DemianNativeV52BSubstrate,
    "demian_native_v52b": DemianNativeV52BSubstrate,
    "demian_native_v5.2c": DemianNativeV52CSubstrate,
    "demian_native_v52c": DemianNativeV52CSubstrate,
    "demian_native_v5.3": DemianNativeV53Substrate,
    "demian_native_v53": DemianNativeV53Substrate,
    "demian_native_v6": DemianNativeV6Substrate,
    "demian_native_v7": DemianNativeV7Substrate,
    "demian_native_v7.1": DemianNativeV71Substrate,
    "demian_native_v71": DemianNativeV71Substrate,
    "demian_native_v7.2": DemianNativeV72Substrate,
    "demian_native_v72": DemianNativeV72Substrate,
    "demian_native_v7.2_surface_ablation": DemianNativeV72SurfaceAblationSubstrate,
    "demian_native_v72_surface_ablation": DemianNativeV72SurfaceAblationSubstrate,
    "demian_native_v7.2_low_exposure": DemianNativeV72LowExposureSubstrate,
    "demian_native_v72_low_exposure": DemianNativeV72LowExposureSubstrate,
    "demian_native_v7.2_refractory": DemianNativeV72RefractorySubstrate,
    "demian_native_v72_refractory": DemianNativeV72RefractorySubstrate,
    "demian_native_v7.2_observer_only": DemianNativeV72ObserverOnlySubstrate,
    "demian_native_v72_observer_only": DemianNativeV72ObserverOnlySubstrate,
    "demian_native_v7.2_tightness_only": DemianNativeV72TightnessOnlySubstrate,
    "demian_native_v72_tightness_only": DemianNativeV72TightnessOnlySubstrate,
    "demian_native_v7.2_constraint_only": DemianNativeV72ConstraintOnlySubstrate,
    "demian_native_v72_constraint_only": DemianNativeV72ConstraintOnlySubstrate,
    "demian_native_v7.2_drive_only": DemianNativeV72DriveOnlySubstrate,
    "demian_native_v72_drive_only": DemianNativeV72DriveOnlySubstrate,
    "demian_native_v7.4": DemianNativeV74Substrate,
    "demian_native_v74": DemianNativeV74Substrate,
    "demian_native_v7.4_pressure_policy": DemianNativeV74PressurePolicySubstrate,
    "demian_native_v74_pressure_policy": DemianNativeV74PressurePolicySubstrate,
    "demian_native_v7.4_calibrated_pressure_policy": DemianNativeV74CalibratedPressurePolicySubstrate,
    "demian_native_v74_calibrated_pressure_policy": DemianNativeV74CalibratedPressurePolicySubstrate,
    "demian_native_v8": DemianNativeV8Substrate,
    "demian_native_v80": DemianNativeV8Substrate,
    "demian_native_v8.5": DemianNativeV85Substrate,
    "demian_native_v85": DemianNativeV85Substrate,
    "demian_native_v9": DemianNativeV9Substrate,
    "demian_native_v90": DemianNativeV9Substrate,
    "lstm": LSTMSubstrate,
    "diag_ssm": DiagonalSSMSubstrate,
    "sel_ssm": SelectiveSSMSubstrate,
}

DUAL_GRU_V3B_REGIMES: Dict[str, Dict[str, float]] = {
    "tight": {
        "message_self_retention": 0.0,
        "slow_carry_scale": 0.0,
        "message_drive_scale": 0.0,
    },
    "threshold": {
        "message_self_retention": 0.36,
        "slow_carry_scale": 0.04,
        "message_drive_scale": 0.09,
    },
    "edge": {
        "message_self_retention": 0.54,
        "slow_carry_scale": 0.05,
        "message_drive_scale": 0.12,
    },
    "current": {},
    "saturated": {
        "message_self_retention": 1.08,
        "slow_carry_scale": 0.12,
        "message_drive_scale": 0.27,
    },
}


def list_substrate_specs() -> List[str]:
    specs = list(SUBSTRATE_REGISTRY.keys())
    specs.extend(f"dual_gru_v3b:{name}" for name in DUAL_GRU_V3B_REGIMES)
    return specs


def _resolve_substrate_spec(
    substrate_name: str,
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> tuple[str, Dict[str, float]]:
    kwargs = dict(substrate_kwargs or {})
    if ":" not in substrate_name:
        return substrate_name, kwargs

    base_name, regime = substrate_name.split(":", 1)
    if base_name != "dual_gru_v3b":
        raise KeyError(f"Unknown substrate spec: {substrate_name}")
    if regime not in DUAL_GRU_V3B_REGIMES:
        raise KeyError(f"Unknown dual_gru_v3b regime: {regime}")

    merged = dict(DUAL_GRU_V3B_REGIMES[regime])
    merged.update(kwargs)
    return base_name, merged


def _make_substrate(
    substrate_name: str,
    hidden_size: int,
    substrate_kwargs: Optional[Dict[str, float]] = None,
):
    canonical_name, kwargs = _resolve_substrate_spec(substrate_name, substrate_kwargs)
    if canonical_name not in SUBSTRATE_REGISTRY:
        raise KeyError(f"Unknown substrate: {substrate_name}")
    return SUBSTRATE_REGISTRY[canonical_name](hidden_size, **kwargs)


class SelfLoopRunner:
    def __init__(
        self,
        substrate: SelfLoopSubstrate,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        window_size: int = 20,
        residual_buffer_size: int = 16,
    ):
        self.substrate = substrate.to(device=device, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.window_size = window_size
        self.residual_buffer_size = residual_buffer_size

    def run(
        self,
        steps: int,
        seed: int,
        perturb_step: Optional[int] = None,
        perturb_scale: float = 0.0,
        perturb_mode: str = "noise",
        perturb_schedule: Optional[Dict[int, float]] = None,
        initial_state: object | None = None,
    ) -> tuple[List[StepMetrics], RunSummary, torch.Tensor]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            if initial_state is None:
                state = self.substrate.initial_state(1, self.device)
            else:
                state = initial_state

            trajectory: List[StepMetrics] = []
            window: List[dict] = []
            residual_buffer: List[torch.Tensor] = []
            prev = None
            prev_vel = None
            prev_components: Optional[Dict[str, torch.Tensor]] = None

            for step_idx in range(1, steps + 1):
                if perturb_schedule and step_idx in perturb_schedule:
                    state = self._apply_perturbation(state, perturb_schedule[step_idx], perturb_mode)
                elif perturb_step is not None and step_idx == perturb_step:
                    state = self._apply_perturbation(state, perturb_scale, perturb_mode)

                state = self.substrate.step(state)
                h = self.substrate.state_vector(state).view(-1).detach().float().cpu()
                components = {
                    name: tensor.view(-1).detach().float().cpu()
                    for name, tensor in self.substrate.state_components(state).items()
                }
                resid_np = h.numpy()

                norm = float(torch.norm(h)) / math.sqrt(max(h.shape[0], 1))
                if prev is not None:
                    dv = h - prev
                    delta = float(torch.norm(dv)) / math.sqrt(max(h.shape[0], 1))
                    coh = float(torch.nn.functional.cosine_similarity(h, prev, dim=0))
                else:
                    dv = torch.zeros_like(h)
                    delta = 0.0
                    coh = 1.0

                if prev_vel is not None:
                    vn = float(prev_vel.norm() * dv.norm())
                    vel = float(torch.dot(prev_vel, dv) / vn) if vn > 1e-10 else 0.0
                else:
                    vel = 1.0

                sc, scon = _fft_spectrum(resid_np)
                fast_vec = components.get("fast", h)
                slow_vec = components.get("slow", torch.zeros(1))
                message_vec = components.get("message", torch.zeros(1))
                if prev_components is None:
                    fast_delta = 0.0
                    slow_delta = 0.0
                    message_delta = 0.0
                    fast_contraction = 1.0
                    slow_contraction = 1.0
                    message_contraction = 1.0
                else:
                    fast_delta = float(torch.norm(fast_vec - prev_components["fast"])) / math.sqrt(max(fast_vec.shape[0], 1))
                    slow_delta = (
                        float(torch.norm(slow_vec - prev_components["slow"])) / math.sqrt(max(slow_vec.shape[0], 1))
                        if "slow" in prev_components and "slow" in components
                        else 0.0
                    )
                    message_delta = (
                        float(torch.norm(message_vec - prev_components["message"])) / math.sqrt(max(message_vec.shape[0], 1))
                        if "message" in prev_components and "message" in components
                        else 0.0
                    )
                    fast_contraction = float(fast_vec.norm() / (prev_components["fast"].norm() + 1e-10))
                    slow_contraction = (
                        float(slow_vec.norm() / (prev_components["slow"].norm() + 1e-10))
                        if "slow" in prev_components and "slow" in components
                        else 1.0
                    )
                    message_contraction = (
                        float(message_vec.norm() / (prev_components["message"].norm() + 1e-10))
                        if "message" in prev_components and "message" in components
                        else 1.0
                    )
                aux = self.substrate.step_aux()
                step_metrics = StepMetrics(
                    step=step_idx,
                    residual_norm=norm,
                    residual_delta=delta,
                    temporal_coherence=coh,
                    velocity_align=vel,
                    spectral_centroid=sc,
                    spectral_concentration=scon,
                    layer_work_ratio=0.5,
                    fast_state_norm=float(torch.norm(fast_vec)) / math.sqrt(max(fast_vec.shape[0], 1)),
                    slow_state_norm=float(torch.norm(slow_vec)) / math.sqrt(max(slow_vec.shape[0], 1)) if "slow" in components else 0.0,
                    message_state_norm=float(torch.norm(message_vec)) / math.sqrt(max(message_vec.shape[0], 1)) if "message" in components else 0.0,
                    fast_state_delta=fast_delta,
                    slow_state_delta=slow_delta,
                    message_state_delta=message_delta,
                    fast_contraction_ratio=fast_contraction,
                    slow_contraction_ratio=slow_contraction,
                    message_contraction_ratio=message_contraction,
                    slow_write_mean=float(aux.get("slow_write_mean", 0.0)),
                    message_write_mean=float(aux.get("message_write_mean", 0.0)),
                    fast_update_mean=float(aux.get("fast_update_mean", 0.0)),
                    route_metrics=dict(aux),
                )
                trajectory.append(step_metrics)
                window.append(asdict(step_metrics))
                if len(window) > self.window_size:
                    window.pop(0)
                residual_buffer.append(h.clone())
                if len(residual_buffer) > self.residual_buffer_size:
                    residual_buffer.pop(0)

                prev = h.clone()
                prev_vel = dv.clone()
                prev_components = {name: tensor.clone() for name, tensor in components.items()}

            obs = compute_observables(window, prev, residual_buffer)
            attractor = classify_attractor(obs)
            summary = RunSummary(
                substrate=self._substrate_name(),
                seed=seed,
                steps=steps,
                perturb_step=perturb_step,
                perturb_scale=perturb_scale,
                final_norm=trajectory[-1].residual_norm,
                mean_norm=float(np.mean([s.residual_norm for s in trajectory])),
                std_norm=float(np.std([s.residual_norm for s in trajectory])),
                mean_delta=float(np.mean([s.residual_delta for s in trajectory])),
                mean_coherence=float(np.mean([s.temporal_coherence for s in trajectory])),
                mean_velocity_align=float(np.mean([s.velocity_align for s in trajectory])),
                cycle_period=float(obs["cycle_period"]),
                two_cycle_amplitude=float(obs["two_cycle_amplitude"]),
                covariance_rank=float(obs["covariance_rank"]),
                flow_dimension=float(obs["flow_dimension"]),
                compression_ratio=float(obs["compression_ratio"]),
                attractor_type=attractor.type,
                interior_class="",
                attractor_confidence=float(attractor.stability),
                final_state_checksum=float(prev[: min(8, prev.shape[0])].sum().item()),
                mean_fast_norm=float(np.mean([s.fast_state_norm for s in trajectory])),
                mean_slow_norm=float(np.mean([s.slow_state_norm for s in trajectory])),
                mean_message_norm=float(np.mean([s.message_state_norm for s in trajectory])),
                max_fast_norm=float(np.max([s.fast_state_norm for s in trajectory])),
                max_slow_norm=float(np.max([s.slow_state_norm for s in trajectory])),
                max_message_norm=float(np.max([s.message_state_norm for s in trajectory])),
                mean_fast_delta=float(np.mean([s.fast_state_delta for s in trajectory])),
                mean_slow_delta=float(np.mean([s.slow_state_delta for s in trajectory])),
                mean_message_delta=float(np.mean([s.message_state_delta for s in trajectory])),
                mean_fast_contraction=float(np.mean([s.fast_contraction_ratio for s in trajectory])),
                mean_slow_contraction=float(np.mean([s.slow_contraction_ratio for s in trajectory])),
                mean_message_contraction=float(np.mean([s.message_contraction_ratio for s in trajectory])),
                max_fast_contraction=float(np.max([s.fast_contraction_ratio for s in trajectory])),
                max_slow_contraction=float(np.max([s.slow_contraction_ratio for s in trajectory])),
                max_message_contraction=float(np.max([s.message_contraction_ratio for s in trajectory])),
                slow_fast_delta_ratio=float(
                    np.mean([s.slow_state_delta for s in trajectory]) / (np.mean([s.fast_state_delta for s in trajectory]) + 1e-10)
                ),
                mean_slow_write=float(np.mean([s.slow_write_mean for s in trajectory])),
                mean_message_write=float(np.mean([s.message_write_mean for s in trajectory])),
                mean_fast_update=float(np.mean([s.fast_update_mean for s in trajectory])),
            )
            summary.interior_class = classify_fixed_point_interior(summary)
            return trajectory, summary, prev

    def _apply_perturbation(self, state: object, scale: float, mode: str = "noise") -> object:
        if scale <= 0:
            return state
        if mode in {"rss_negation", "surface_negation"}:
            surface = self.substrate.state_vector(state).to(device=self.device, dtype=self.dtype)
            negated = (1.0 - scale) * surface - scale * surface
            return self.substrate.write_surface_state(state, negated)
        if mode != "noise":
            raise ValueError(f"Unknown perturbation mode: {mode}")
        if isinstance(state, tuple):
            return tuple(s + scale * torch.randn_like(s) for s in state)
        return state + scale * torch.randn_like(state)

    def _substrate_name(self) -> str:
        for name, cls in SUBSTRATE_REGISTRY.items():
            if type(self.substrate) is cls:
                return name
        for name, cls in SUBSTRATE_REGISTRY.items():
            if isinstance(self.substrate, cls):
                return name
        return self.substrate.__class__.__name__.lower()


def basin_map(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> List[RunSummary]:
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    summaries = []
    for seed in seeds:
        _, summary, _ = runner.run(steps=steps, seed=seed)
        summary.substrate = substrate_name
        summaries.append(summary)
    return summaries


def perturbation_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    perturb_mode: str = "noise",
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)

    base_state = model.initial_state(1, runner.device)
    clean_traj, clean_summary, clean_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )
    pert_traj, pert_summary, pert_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        perturb_mode=perturb_mode,
    )

    final_cos = float(
        torch.nn.functional.cosine_similarity(clean_final, pert_final, dim=0).item()
    )
    norm_gap = float(torch.norm(clean_final - pert_final).item())
    return {
        "clean": asdict(clean_summary),
        "perturbed": asdict(pert_summary),
        "perturb_mode": perturb_mode,
        "final_cosine": final_cos,
        "final_l2_gap": norm_gap,
        "recovery_window_mean_gap": float(
            np.mean(
                [
                    abs(a.residual_norm - b.residual_norm)
                    for a, b in zip(clean_traj[max(perturb_step - 1, 0):], pert_traj[max(perturb_step - 1, 0):])
                ]
            )
        ),
    }


def _clone_substrate_state(state: object) -> object:
    if isinstance(state, tuple):
        return tuple(_clone_substrate_state(item) for item in state)
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    return state


def _manual_step_trace(
    model: SelfLoopSubstrate,
    state: object,
    steps: int,
) -> tuple[object, list[torch.Tensor], list[Dict[str, float]]]:
    vectors: list[torch.Tensor] = []
    route_metrics: list[Dict[str, float]] = []
    with torch.no_grad():
        for _ in range(steps):
            state = model.step(state)
            vectors.append(model.state_vector(state).view(-1).detach().float().cpu())
            route_metrics.append(dict(model.step_aux()))
    return state, vectors, route_metrics


def _mean_abs_metric_gap(
    baseline: list[Dict[str, float]],
    candidate: list[Dict[str, float]],
    names: list[str],
) -> Dict[str, float]:
    gaps: Dict[str, float] = {}
    for name in names:
        values = [
            abs(float(a.get(name, 0.0)) - float(b.get(name, 0.0)))
            for a, b in zip(baseline, candidate)
        ]
        gaps[name] = float(np.mean(values)) if values else 0.0
    return gaps


def resume_continuity_probe(
    substrate_name: str = "demian_native_v7.4",
    hidden_size: int = 32,
    seed: int = 94,
    pause_steps: int = 128,
    resume_steps: int = 128,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compare continuity capsule resume with component-lesioned reconstructions."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    source = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    source = source.to(device=torch.device(device), dtype=torch.float32)
    pause_state = source.initial_state(1, torch.device(device))
    pause_state, pre_vectors, pre_metrics = _manual_step_trace(source, pause_state, pause_steps)
    pause_surface = source.state_vector(pause_state).detach().clone()
    pause_body = {
        name: value.detach().clone()
        for name, value in source.state_dict().items()
    }

    uninterrupted_state, uninterrupted_vectors, uninterrupted_metrics = _manual_step_trace(
        source,
        _clone_substrate_state(pause_state),
        resume_steps,
    )

    def final_cos(vectors: list[torch.Tensor]) -> float:
        if not uninterrupted_vectors or not vectors:
            return 0.0
        return float(torch.nn.functional.cosine_similarity(uninterrupted_vectors[-1], vectors[-1], dim=0).item())

    def mean_gap(vectors: list[torch.Tensor]) -> float:
        gaps = [
            float(torch.norm(a - b)) / math.sqrt(max(a.shape[0], 1))
            for a, b in zip(uninterrupted_vectors, vectors)
        ]
        return float(np.mean(gaps)) if gaps else 0.0

    def final_gap(vectors: list[torch.Tensor]) -> float:
        if not uninterrupted_vectors or not vectors:
            return 0.0
        return float(torch.norm(uninterrupted_vectors[-1] - vectors[-1]))

    route_names = [
        "v74_viability_mean",
        "v74_ownership_mean",
        "v74_tension_mean",
        "v74_hold_pressure_mean",
        "v74_resolution_open_mean",
        "v74_topology_state_norm",
        "v74_dynamic_topology_injection_norm",
        "topology_state_norm",
    ]

    def run_arm(
        *,
        restore_body: bool,
        restore_state: bool,
        restore_surface: bool,
    ) -> Dict[str, object]:
        torch.manual_seed(seed)
        model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
        model = model.to(device=torch.device(device), dtype=torch.float32)
        if restore_body:
            model.load_state_dict(pause_body)
        if restore_state:
            arm_state = _clone_substrate_state(pause_state)
        else:
            arm_state = model.initial_state(1, torch.device(device))
        if restore_surface:
            arm_state = model.write_surface_state(arm_state, pause_surface)
        arm_state, vectors, metrics = _manual_step_trace(model, arm_state, resume_steps)
        arm_final_cosine = final_cos(vectors)
        arm_mean_gap = mean_gap(vectors)
        return {
            "final_cosine_vs_uninterrupted": arm_final_cosine,
            "final_l2_gap_vs_uninterrupted": final_gap(vectors),
            "mean_step_gap_vs_uninterrupted": arm_mean_gap,
            "route_metric_mean_abs_gap": _mean_abs_metric_gap(uninterrupted_metrics, metrics, route_names),
        }

    capsule_resume = run_arm(restore_body=True, restore_state=True, restore_surface=False)
    state_only_resume = run_arm(restore_body=False, restore_state=True, restore_surface=False)
    body_only_resume = run_arm(restore_body=True, restore_state=False, restore_surface=False)
    notebook_resume = run_arm(restore_body=False, restore_state=False, restore_surface=True)
    body_notebook_resume = run_arm(restore_body=True, restore_state=False, restore_surface=True)

    return {
        "substrate": substrate_name,
        "config": {
            "hidden_size": hidden_size,
            "seed": seed,
            "pause_steps": pause_steps,
            "resume_steps": resume_steps,
            "device": device,
            "substrate_kwargs": dict(substrate_kwargs or {}),
        },
        "pause": {
            "surface_norm": float(torch.norm(pause_surface).item()),
            "last_route_metrics": pre_metrics[-1] if pre_metrics else {},
        },
        "capsule_resume": capsule_resume,
        "state_only_resume": state_only_resume,
        "body_only_resume": body_only_resume,
        "notebook_resume": notebook_resume,
        "body_notebook_resume": body_notebook_resume,
        "continuity_advantage": {
            "final_cosine": (
                float(capsule_resume["final_cosine_vs_uninterrupted"])
                - float(notebook_resume["final_cosine_vs_uninterrupted"])
            ),
            "mean_step_gap": (
                float(notebook_resume["mean_step_gap_vs_uninterrupted"])
                - float(capsule_resume["mean_step_gap_vs_uninterrupted"])
            ),
            "trajectory_shape_ratio": (
                float(notebook_resume["mean_step_gap_vs_uninterrupted"])
                / (float(capsule_resume["mean_step_gap_vs_uninterrupted"]) + 1e-10)
            ),
        },
        "component_advantage": {
            "state_only_gap_ratio": (
                float(state_only_resume["mean_step_gap_vs_uninterrupted"])
                / (float(capsule_resume["mean_step_gap_vs_uninterrupted"]) + 1e-10)
            ),
            "body_only_gap_ratio": (
                float(body_only_resume["mean_step_gap_vs_uninterrupted"])
                / (float(capsule_resume["mean_step_gap_vs_uninterrupted"]) + 1e-10)
            ),
            "body_notebook_gap_ratio": (
                float(body_notebook_resume["mean_step_gap_vs_uninterrupted"])
                / (float(capsule_resume["mean_step_gap_vs_uninterrupted"]) + 1e-10)
            ),
        },
    }


def memory_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    initial_delta: float,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)

    base_state = model.initial_state(1, runner.device)
    if isinstance(base_state, tuple):
        alt_state = tuple(
            s + initial_delta * torch.randn_like(s) for s in base_state
        )
    else:
        alt_state = base_state + initial_delta * torch.randn_like(base_state)

    _, clean_summary, clean_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )
    _, alt_summary, alt_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=alt_state,
    )
    return {
        "base": asdict(clean_summary),
        "alt": asdict(alt_summary),
        "final_cosine": float(torch.nn.functional.cosine_similarity(clean_final, alt_final, dim=0).item()),
        "final_l2_gap": float(torch.norm(clean_final - alt_final).item()),
    }


def bottleneck_run(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    bottleneck_dim: int,
    bottleneck_interval: int,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    proj = _fixed_projection(hidden_size, bottleneck_dim, seed + 101)

    base_state = model.initial_state(1, runner.device)
    _, clean_summary, clean_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )

    with torch.no_grad():
        state = base_state
        codes: List[torch.Tensor] = []
        prev = None
        deltas: List[float] = []
        for step_idx in range(1, steps + 1):
            state = model.step(state)
            h = model.state_vector(state).view(-1).detach().float().cpu()
            if step_idx % bottleneck_interval == 0:
                code = _encode_state(h, proj)
                codes.append(code)
                recon = _decode_code(code, proj).to(runner.device, dtype=runner.dtype)
                recon = recon * (h.norm() + 1e-10)
                state = model.write_surface_state(state, recon)
            if prev is not None:
                deltas.append(float(torch.norm(h - prev)))
            prev = h
        bottleneck_final = model.state_vector(state).view(-1).detach().float().cpu()

    stats = _code_stats(codes)
    final_cos = float(torch.nn.functional.cosine_similarity(clean_final, bottleneck_final, dim=0).item())
    return {
        "clean": asdict(clean_summary),
        "final_cosine_vs_clean": final_cos,
        "final_l2_gap_vs_clean": float(torch.norm(clean_final - bottleneck_final).item()),
        "mean_step_delta_after_bottleneck": float(np.mean(deltas)) if deltas else 0.0,
        "bottleneck_dim": bottleneck_dim,
        "bottleneck_interval": bottleneck_interval,
        **stats,
    }


def coupled_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    canonical_name, kwargs = _resolve_substrate_spec(substrate_name, substrate_kwargs)
    model_a = _make_substrate(canonical_name, hidden_size, kwargs)
    model_b = _make_substrate(canonical_name, hidden_size, kwargs)
    runner_a = SelfLoopRunner(model_a, device=device)
    runner_b = SelfLoopRunner(model_b, device=device)
    proj = _fixed_projection(hidden_size, coupling_dim, seed + 211)

    state_a = model_a.initial_state(1, runner_a.device)
    state_b = model_b.initial_state(1, runner_b.device)
    cosines: List[float] = []
    codes_a: List[torch.Tensor] = []
    codes_b: List[torch.Tensor] = []

    with torch.no_grad():
        for step_idx in range(1, steps + 1):
            state_a = model_a.step(state_a)
            state_b = model_b.step(state_b)
            h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
            h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()

            if step_idx % coupling_interval == 0:
                code_a = _encode_state(h_a, proj)
                code_b = _encode_state(h_b, proj)
                codes_a.append(code_a)
                codes_b.append(code_b)
                msg_a = _decode_code(code_a, proj).to(runner_a.device, dtype=runner_a.dtype)
                msg_b = _decode_code(code_b, proj).to(runner_b.device, dtype=runner_b.dtype)
                state_a = model_a.inject_coupling_message(state_a, msg_b, coupling_strength)
                state_b = model_b.inject_coupling_message(state_b, msg_a, coupling_strength)

                h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
                h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()

            cosines.append(float(torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item()))

    return {
        "initial_cosine": cosines[0] if cosines else 0.0,
        "final_cosine": cosines[-1] if cosines else 0.0,
        "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
        "coupling_strength": coupling_strength,
        "coupling_dim": coupling_dim,
        "coupling_interval": coupling_interval,
        "agent_a": _code_stats(codes_a),
        "agent_b": _code_stats(codes_b),
    }


def perturbation_stress_test(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scales: List[float],
    perturb_mode: str = "noise",
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    base_state = model.initial_state(1, runner.device)

    _, baseline_summary, baseline_final = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )

    cases = []
    for scale in perturb_scales:
        _, summary, final_state = runner.run(
            steps=steps,
            seed=seed,
            initial_state=base_state,
            perturb_step=perturb_step,
            perturb_scale=scale,
            perturb_mode=perturb_mode,
        )
        cases.append(
            {
                "perturb_scale": scale,
                "perturb_mode": perturb_mode,
                "summary": asdict(summary),
                "final_cosine_vs_baseline": float(
                    torch.nn.functional.cosine_similarity(baseline_final, final_state, dim=0).item()
                ),
                "peak_norm_ratio_vs_baseline": {
                    "fast": float(summary.max_fast_norm / (baseline_summary.max_fast_norm + 1e-10)),
                    "slow": float(summary.max_slow_norm / (baseline_summary.max_slow_norm + 1e-10)),
                    "message": float(summary.max_message_norm / (baseline_summary.max_message_norm + 1e-10)),
                },
                "peak_contraction_ratio": {
                    "fast": float(summary.max_fast_contraction),
                    "slow": float(summary.max_slow_contraction),
                    "message": float(summary.max_message_contraction),
                },
            }
        )

    return {
        "baseline": asdict(baseline_summary),
        "perturb_mode": perturb_mode,
        "cases": cases,
    }


def coupling_stress_test(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strengths: List[float],
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    canonical_name, kwargs = _resolve_substrate_spec(substrate_name, substrate_kwargs)
    results = []

    for strength in coupling_strengths:
        model_a = _make_substrate(canonical_name, hidden_size, kwargs)
        model_b = _make_substrate(canonical_name, hidden_size, kwargs)
        runner_a = SelfLoopRunner(model_a, device=device)
        runner_b = SelfLoopRunner(model_b, device=device)
        proj = _fixed_projection(hidden_size, coupling_dim, seed + 911)

        state_a = model_a.initial_state(1, runner_a.device)
        state_b = model_b.initial_state(1, runner_b.device)
        cosines: List[float] = []
        peak = {"fast": 0.0, "slow": 0.0, "message": 0.0}

        with torch.no_grad():
            for step_idx in range(1, steps + 1):
                state_a = model_a.step(state_a)
                state_b = model_b.step(state_b)
                h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
                h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()

                comps_a = {
                    name: tensor.view(-1).detach().float().cpu()
                    for name, tensor in model_a.state_components(state_a).items()
                }
                for name in peak:
                    if name in comps_a:
                        peak[name] = max(
                            peak[name],
                            float(torch.norm(comps_a[name])) / math.sqrt(max(comps_a[name].shape[0], 1)),
                        )

                if step_idx % coupling_interval == 0:
                    code_a = _encode_state(h_a, proj)
                    code_b = _encode_state(h_b, proj)
                    msg_a = _decode_code(code_a, proj).to(runner_a.device, dtype=runner_a.dtype)
                    msg_b = _decode_code(code_b, proj).to(runner_b.device, dtype=runner_b.dtype)
                    state_a = model_a.inject_coupling_message(state_a, msg_b, strength)
                    state_b = model_b.inject_coupling_message(state_b, msg_a, strength)
                    h_a = model_a.state_vector(state_a).view(-1).detach().float().cpu()
                    h_b = model_b.state_vector(state_b).view(-1).detach().float().cpu()

                cosines.append(float(torch.nn.functional.cosine_similarity(h_a, h_b, dim=0).item()))

        results.append(
            {
                "coupling_strength": strength,
                "initial_cosine": cosines[0] if cosines else 0.0,
                "final_cosine": cosines[-1] if cosines else 0.0,
                "mean_cosine": float(np.mean(cosines)) if cosines else 0.0,
                "peak_component_norms": peak,
            }
        )

    return {"cases": results}


def _entry_markers(
    trajectory: List[StepMetrics],
    message_norm_factor: float = 2.0,
    message_contraction_threshold: float = 1.02,
) -> Dict[str, Optional[int]]:
    if not trajectory:
        return {
            "message_norm_takeoff_step": None,
            "message_contraction_takeoff_step": None,
            "slow_norm_takeoff_step": None,
        }

    base_message_norm = trajectory[0].message_state_norm
    base_slow_norm = trajectory[0].slow_state_norm
    msg_norm_step = None
    msg_contr_step = None
    slow_norm_step = None

    for step in trajectory:
        if (
            msg_norm_step is None
            and base_message_norm > 1e-10
            and step.message_state_norm >= message_norm_factor * base_message_norm
        ):
            msg_norm_step = step.step
        if (
            msg_contr_step is None
            and step.message_contraction_ratio >= message_contraction_threshold
        ):
            msg_contr_step = step.step
        if (
            slow_norm_step is None
            and base_slow_norm > 1e-10
            and step.slow_state_norm >= message_norm_factor * base_slow_norm
        ):
            slow_norm_step = step.step

    return {
        "message_norm_takeoff_step": msg_norm_step,
        "message_contraction_takeoff_step": msg_contr_step,
        "slow_norm_takeoff_step": slow_norm_step,
    }


def _collapse_onset_step(markers: Dict[str, Optional[int]]) -> Optional[int]:
    candidates = [
        markers.get("message_contraction_takeoff_step"),
        markers.get("message_norm_takeoff_step"),
    ]
    candidates = [int(step) for step in candidates if step is not None]
    return min(candidates) if candidates else None


def early_collapse_window(
    mapping: Dict[str, object],
    state_key: str = "state_triggered",
    window_radius: int = 3,
) -> Dict[str, object]:
    """Extract the local window around the first early-collapse onset.

    The onset is defined structurally: earliest message contraction or message
    norm takeoff. No semantics, only where the message channel first opens away
    from the baseline interior.
    """
    state_payload = mapping.get(state_key, {}) if isinstance(mapping, dict) else {}
    markers = dict(state_payload.get("entry_markers", {}))
    onset_step = _collapse_onset_step(markers)
    trajectory = list(state_payload.get("trajectory", []))
    observer_trace = list(mapping.get("observer_trace", []))
    controller_states = list(mapping.get("controller_states", []))
    trigger_strengths = list(mapping.get("trigger_strengths", []))
    burst_trace = list(mapping.get("burst_trace", []))
    refractory_trace = list(mapping.get("refractory_trace", []))
    burst_open_trace = list(mapping.get("burst_open_trace", []))
    trigger_steps = set(mapping.get("trigger_steps", []))

    if onset_step is None:
        return {
            "onset_step": None,
            "window_start": None,
            "window_end": None,
            "rows": [],
            "markers": markers,
        }

    lo = max(1, onset_step - window_radius)
    hi = min(len(trajectory) if trajectory else onset_step, onset_step + window_radius)
    rows = []
    for step in range(lo, hi + 1):
        row: Dict[str, object] = {"step": step, "triggered": step in trigger_steps}
        if step - 1 < len(trajectory):
            row["trajectory"] = trajectory[step - 1]
        if step - 1 < len(observer_trace):
            row["observer"] = observer_trace[step - 1]
        if step - 1 < len(controller_states):
            row["controller_state"] = float(controller_states[step - 1])
        if step - 1 < len(trigger_strengths):
            row["trigger_strength"] = float(trigger_strengths[step - 1])
        if step - 1 < len(burst_trace):
            row["burst_remaining"] = int(burst_trace[step - 1])
        if step - 1 < len(refractory_trace):
            row["refractory_remaining"] = int(refractory_trace[step - 1])
        if step - 1 < len(burst_open_trace):
            row["burst_open_count"] = int(burst_open_trace[step - 1])
        rows.append(row)

    return {
        "onset_step": onset_step,
        "window_start": lo,
        "window_end": hi,
        "rows": rows,
        "markers": markers,
    }


def classify_fixed_point_interior(summary: RunSummary) -> str:
    if summary.attractor_type != "FIXED_POINT":
        return "non_fixed_point"

    accumulating = (
        summary.max_message_norm >= 1.0
        or summary.max_slow_norm >= 1.0
        or summary.mean_message_contraction >= 1.01
        or summary.max_message_contraction >= 1.05
    )
    if accumulating:
        return "accumulating_fixed_point"
    return "tight_fixed_point"


def trajectory_map(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
    perturb_step: Optional[int] = None,
    perturb_scale: float = 0.0,
    perturb_mode: str = "noise",
    perturb_schedule: Optional[Dict[int, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    trajectory, summary, _ = runner.run(
        steps=steps,
        seed=seed,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        perturb_mode=perturb_mode,
        perturb_schedule=perturb_schedule,
    )

    return {
        "summary": asdict(summary),
        "entry_markers": _entry_markers(trajectory),
        "trajectory": [asdict(step) for step in trajectory],
    }


def trajectory_map_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    perturb_mode: str = "noise",
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    base_state = model.initial_state(1, runner.device)

    clean_traj, clean_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )
    pert_traj, pert_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        perturb_mode=perturb_mode,
    )

    return {
        "clean": {
            "summary": asdict(clean_summary),
            "entry_markers": _entry_markers(clean_traj),
            "trajectory": [asdict(step) for step in clean_traj],
        },
        "perturbed": {
            "summary": asdict(pert_summary),
            "perturb_mode": perturb_mode,
            "entry_markers": _entry_markers(pert_traj),
            "trajectory": [asdict(step) for step in pert_traj],
        },
    }


def trajectory_map_schedule_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_schedule: Dict[int, float],
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    model = _make_substrate(substrate_name, hidden_size, substrate_kwargs)
    runner = SelfLoopRunner(model, device=device)
    base_state = model.initial_state(1, runner.device)

    clean_traj, clean_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )
    pert_traj, pert_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
        perturb_schedule=perturb_schedule,
    )

    return {
        "clean": {
            "summary": asdict(clean_summary),
            "entry_markers": _entry_markers(clean_traj),
            "trajectory": [asdict(step) for step in clean_traj],
        },
        "perturbed": {
            "summary": asdict(pert_summary),
            "entry_markers": _entry_markers(pert_traj),
            "trajectory": [asdict(step) for step in pert_traj],
        },
        "perturb_schedule": perturb_schedule,
    }


def self_coupling_schedule_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    trigger_schedule: Dict[int, float],
    coupling_dim: int = 8,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    canonical_name, kwargs = _resolve_substrate_spec(substrate_name, substrate_kwargs)
    model = _make_substrate(canonical_name, hidden_size, kwargs)
    runner = SelfLoopRunner(model, device=device)
    proj = _fixed_projection(hidden_size, coupling_dim, seed + 307)
    base_state = model.initial_state(1, runner.device)

    clean_traj, clean_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )

    with torch.no_grad():
        state = base_state
        trajectory: List[StepMetrics] = []
        window: List[Dict[str, float]] = []
        residual_buffer: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None
        prev_vel: Optional[torch.Tensor] = None
        prev_components: Optional[Dict[str, torch.Tensor]] = None

        for step_idx in range(1, steps + 1):
            state = model.step(state)
            if step_idx in trigger_schedule:
                h_live = model.state_vector(state).view(-1).detach().float().cpu()
                code = _encode_state(h_live, proj)
                trigger = _decode_code(code, proj).to(runner.device, dtype=runner.dtype)
                state = model.inject_coupling_message(state, trigger, trigger_schedule[step_idx])

            h = model.state_vector(state).view(-1).detach().float().cpu()
            components = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            resid_np = h.numpy()

            norm = float(torch.norm(h)) / math.sqrt(max(h.shape[0], 1))
            if prev is not None:
                dv = h - prev
                delta = float(torch.norm(dv)) / math.sqrt(max(h.shape[0], 1))
                coh = float(torch.nn.functional.cosine_similarity(h, prev, dim=0))
            else:
                dv = torch.zeros_like(h)
                delta = 0.0
                coh = 1.0

            if prev_vel is not None:
                vn = float(prev_vel.norm() * dv.norm())
                vel = float(torch.dot(prev_vel, dv) / vn) if vn > 1e-10 else 0.0
            else:
                vel = 1.0

            sc, scon = _fft_spectrum(resid_np)
            fast_vec = components.get("fast", h)
            slow_vec = components.get("slow", torch.zeros(1))
            message_vec = components.get("message", torch.zeros(1))
            if prev_components is None:
                fast_delta = 0.0
                slow_delta = 0.0
                message_delta = 0.0
                fast_contraction = 1.0
                slow_contraction = 1.0
                message_contraction = 1.0
            else:
                fast_delta = float(torch.norm(fast_vec - prev_components["fast"])) / math.sqrt(max(fast_vec.shape[0], 1))
                slow_delta = (
                    float(torch.norm(slow_vec - prev_components["slow"])) / math.sqrt(max(slow_vec.shape[0], 1))
                    if "slow" in prev_components and "slow" in components
                    else 0.0
                )
                message_delta = (
                    float(torch.norm(message_vec - prev_components["message"])) / math.sqrt(max(message_vec.shape[0], 1))
                    if "message" in prev_components and "message" in components
                    else 0.0
                )
                fast_contraction = float(fast_vec.norm() / (prev_components["fast"].norm() + 1e-10))
                slow_contraction = (
                    float(slow_vec.norm() / (prev_components["slow"].norm() + 1e-10))
                    if "slow" in prev_components and "slow" in components
                    else 1.0
                )
                message_contraction = (
                    float(message_vec.norm() / (prev_components["message"].norm() + 1e-10))
                    if "message" in prev_components and "message" in components
                    else 1.0
                )

            aux = model.step_aux()
            step_metrics = StepMetrics(
                step=step_idx,
                residual_norm=norm,
                residual_delta=delta,
                temporal_coherence=coh,
                velocity_align=vel,
                spectral_centroid=sc,
                spectral_concentration=scon,
                layer_work_ratio=0.5,
                fast_state_norm=float(torch.norm(fast_vec)) / math.sqrt(max(fast_vec.shape[0], 1)),
                slow_state_norm=float(torch.norm(slow_vec)) / math.sqrt(max(slow_vec.shape[0], 1)) if "slow" in components else 0.0,
                message_state_norm=float(torch.norm(message_vec)) / math.sqrt(max(message_vec.shape[0], 1)) if "message" in components else 0.0,
                fast_state_delta=fast_delta,
                slow_state_delta=slow_delta,
                message_state_delta=message_delta,
                fast_contraction_ratio=fast_contraction,
                slow_contraction_ratio=slow_contraction,
                message_contraction_ratio=message_contraction,
                slow_write_mean=float(aux.get("slow_write_mean", 0.0)),
                message_write_mean=float(aux.get("message_write_mean", 0.0)),
                fast_update_mean=float(aux.get("fast_update_mean", 0.0)),
                route_metrics=dict(aux),
            )
            trajectory.append(step_metrics)
            window.append(asdict(step_metrics))
            if len(window) > runner.window_size:
                window.pop(0)
            residual_buffer.append(h.clone())
            if len(residual_buffer) > runner.residual_buffer_size:
                residual_buffer.pop(0)

            prev = h.clone()
            prev_vel = dv.clone()
            prev_components = {name: tensor.clone() for name, tensor in components.items()}

    obs = compute_observables(window, prev, residual_buffer)
    attractor = classify_attractor(obs)
    summary = RunSummary(
        substrate=canonical_name,
        seed=seed,
        steps=steps,
        perturb_step=None,
        perturb_scale=0.0,
        final_norm=trajectory[-1].residual_norm,
        mean_norm=float(np.mean([s.residual_norm for s in trajectory])),
        std_norm=float(np.std([s.residual_norm for s in trajectory])),
        mean_delta=float(np.mean([s.residual_delta for s in trajectory])),
        mean_coherence=float(np.mean([s.temporal_coherence for s in trajectory])),
        mean_velocity_align=float(np.mean([s.velocity_align for s in trajectory])),
        cycle_period=float(obs["cycle_period"]),
        two_cycle_amplitude=float(obs["two_cycle_amplitude"]),
        covariance_rank=float(obs["covariance_rank"]),
        flow_dimension=float(obs["flow_dimension"]),
        compression_ratio=float(obs["compression_ratio"]),
        attractor_type=attractor.type,
        interior_class="",
        attractor_confidence=float(attractor.stability),
        final_state_checksum=float(prev[: min(8, prev.shape[0])].sum().item()),
        mean_fast_norm=float(np.mean([s.fast_state_norm for s in trajectory])),
        mean_slow_norm=float(np.mean([s.slow_state_norm for s in trajectory])),
        mean_message_norm=float(np.mean([s.message_state_norm for s in trajectory])),
        max_fast_norm=float(np.max([s.fast_state_norm for s in trajectory])),
        max_slow_norm=float(np.max([s.slow_state_norm for s in trajectory])),
        max_message_norm=float(np.max([s.message_state_norm for s in trajectory])),
        mean_fast_delta=float(np.mean([s.fast_state_delta for s in trajectory])),
        mean_slow_delta=float(np.mean([s.slow_state_delta for s in trajectory])),
        mean_message_delta=float(np.mean([s.message_state_delta for s in trajectory])),
        mean_fast_contraction=float(np.mean([s.fast_contraction_ratio for s in trajectory])),
        mean_slow_contraction=float(np.mean([s.slow_contraction_ratio for s in trajectory])),
        mean_message_contraction=float(np.mean([s.message_contraction_ratio for s in trajectory])),
        max_fast_contraction=float(np.max([s.fast_contraction_ratio for s in trajectory])),
        max_slow_contraction=float(np.max([s.slow_contraction_ratio for s in trajectory])),
        max_message_contraction=float(np.max([s.message_contraction_ratio for s in trajectory])),
        slow_fast_delta_ratio=float(
            np.mean([s.slow_state_delta for s in trajectory]) / (np.mean([s.fast_state_delta for s in trajectory]) + 1e-10)
        ),
        mean_slow_write=float(np.mean([s.slow_write_mean for s in trajectory])),
        mean_message_write=float(np.mean([s.message_write_mean for s in trajectory])),
        mean_fast_update=float(np.mean([s.fast_update_mean for s in trajectory])),
    )
    summary.interior_class = classify_fixed_point_interior(summary)

    return {
        "clean": {
            "summary": asdict(clean_summary),
            "entry_markers": _entry_markers(clean_traj),
            "trajectory": [asdict(step) for step in clean_traj],
        },
        "self_triggered": {
            "summary": asdict(summary),
            "entry_markers": _entry_markers(trajectory),
            "trajectory": [asdict(step) for step in trajectory],
        },
        "trigger_schedule": trigger_schedule,
        "coupling_dim": coupling_dim,
    }


def search_dual_gru_v3b_internal_trigger_controller(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    center_step: int,
    pulse_radii: List[int],
    pulse_strides: List[int],
    pulse_strengths: List[float],
    coupling_dims: List[int],
    objective: str = "induce",
    scan_offsets: bool = True,
    device: str = "cpu",
) -> Dict[str, object]:
    if objective not in {"induce", "suppress"}:
        raise ValueError(f"unsupported objective: {objective}")

    cases: List[Dict[str, object]] = []

    for pulse_radius in pulse_radii:
        for pulse_stride in pulse_strides:
            offsets = range(pulse_stride) if scan_offsets else [0]
            for offset in offsets:
                lo = max(1, center_step - pulse_radius + offset)
                hi = min(steps, center_step + pulse_radius)
                schedule_steps = list(range(lo, hi + 1, max(1, pulse_stride)))
                if not schedule_steps:
                    continue
                for pulse_strength in pulse_strengths:
                    schedule = {step: pulse_strength for step in schedule_steps}
                    for coupling_dim in coupling_dims:
                        mapping = self_coupling_schedule_pair(
                            substrate_name=substrate_name,
                            hidden_size=hidden_size,
                            steps=steps,
                            seed=seed,
                            trigger_schedule=schedule,
                            coupling_dim=coupling_dim,
                            device=device,
                        )
                        clean = mapping["clean"]
                        trig = mapping["self_triggered"]
                        clean_summary = clean["summary"]
                        trig_summary = trig["summary"]
                        clean_markers = clean["entry_markers"]
                        trig_markers = trig["entry_markers"]
                        slow_takeoff = trig_markers.get("slow_norm_takeoff_step")
                        peak_message_ratio = float(
                            trig_summary["max_message_norm"] / (clean_summary["max_message_norm"] + 1e-10)
                        )
                        mean_delta_ratio = float(
                            trig_summary["mean_delta"] / (clean_summary["mean_delta"] + 1e-10)
                        )
                        if objective == "induce":
                            objective_score = (
                                (steps - float(slow_takeoff)) if slow_takeoff is not None else -float(steps)
                            ) + 10.0 * peak_message_ratio + 5.0 * mean_delta_ratio
                        else:
                            objective_score = (
                                float(slow_takeoff) if slow_takeoff is not None else 2.0 * float(steps)
                            ) - 10.0 * peak_message_ratio - 5.0 * mean_delta_ratio

                        cases.append(
                            {
                                "pulse_radius": pulse_radius,
                                "pulse_stride": pulse_stride,
                                "offset": offset,
                                "pulse_strength": pulse_strength,
                                "coupling_dim": coupling_dim,
                                "schedule": schedule,
                                "clean_entry_markers": clean_markers,
                                "self_triggered_entry_markers": trig_markers,
                                "peak_message_ratio": peak_message_ratio,
                                "mean_delta_ratio": mean_delta_ratio,
                                "objective": objective,
                                "objective_score": objective_score,
                            }
                        )

    reverse = objective == "induce"
    cases.sort(key=lambda row: row["objective_score"], reverse=reverse)

    return {
        "substrate": substrate_name,
        "seed": seed,
        "steps": steps,
        "hidden_size": hidden_size,
        "center_step": center_step,
        "objective": objective,
        "pulse_radii": pulse_radii,
        "pulse_strides": pulse_strides,
        "pulse_strengths": pulse_strengths,
        "coupling_dims": coupling_dims,
        "scan_offsets": scan_offsets,
        "best_case": cases[0] if cases else None,
        "top_cases": cases[:10],
        "cases": cases,
    }


def _controller_strength_from_params(
    step_idx: int,
    steps: int,
    period: int,
    components: Dict[str, torch.Tensor],
    prev_components: Optional[Dict[str, torch.Tensor]],
    controller_state: float,
    burst_remaining: int,
    refractory_remaining: int,
    burst_open_count: int,
    params: Dict[str, float],
) -> tuple[float, float, int, int, int, Dict[str, float]]:
    fast = components.get("fast", torch.zeros(1))
    slow = components.get("slow", torch.zeros(1))
    message = components.get("message", torch.zeros(1))
    fast_norm = float(torch.norm(fast)) / math.sqrt(max(fast.shape[0], 1))
    slow_norm = float(torch.norm(slow)) / math.sqrt(max(slow.shape[0], 1))
    message_norm = float(torch.norm(message)) / math.sqrt(max(message.shape[0], 1))
    if prev_components is None:
        fast_delta = 0.0
        message_delta = 0.0
        slow_delta = 0.0
    else:
        prev_fast = prev_components.get("fast", torch.zeros_like(fast))
        prev_message = prev_components.get("message", torch.zeros_like(message))
        prev_slow = prev_components.get("slow", torch.zeros_like(slow))
        fast_delta = float(torch.norm(fast - prev_fast)) / math.sqrt(max(fast.shape[0], 1))
        message_delta = float(torch.norm(message - prev_message)) / math.sqrt(max(message.shape[0], 1))
        slow_delta = float(torch.norm(slow - prev_slow)) / math.sqrt(max(slow.shape[0], 1))
    message_slow_tension = abs(message_norm - slow_norm)
    message_slow_ratio = message_norm / (slow_norm + 1e-6)
    slow_growth_pressure = slow_norm + 4.0 * slow_delta
    slow_message_gap = slow_growth_pressure - message_norm
    boundary_pressure = max(message_norm - slow_norm, 0.0) * (slow_delta + 1e-6)
    delta_skew = slow_delta - message_delta
    observer_trace = {
        "fast_norm": fast_norm,
        "slow_norm": slow_norm,
        "message_norm": message_norm,
        "fast_delta": fast_delta,
        "slow_delta": slow_delta,
        "message_delta": message_delta,
        "message_slow_tension": message_slow_tension,
        "message_slow_ratio": message_slow_ratio,
        "slow_growth_pressure": slow_growth_pressure,
        "slow_message_gap": slow_message_gap,
        "boundary_pressure": boundary_pressure,
        "delta_skew": delta_skew,
    }

    theta = 2.0 * math.pi * float(step_idx) / float(max(period, 1))
    phase_sin = math.sin(theta)
    phase_cos = math.cos(theta)
    center = float(params.get("center_step", steps // 2))
    width = max(float(params.get("window_width", 8.0)), 1.0)
    window = math.exp(-((float(step_idx) - center) ** 2) / (2.0 * width * width))
    raw = (
        params.get("bias", 0.0)
        + params.get("phase_sin", 0.0) * phase_sin
        + params.get("phase_cos", 0.0) * phase_cos
        + params.get("window_gain", 0.0) * window
        + params.get("fast_norm_gain", 0.0) * fast_norm
        + params.get("slow_norm_gain", 0.0) * slow_norm
        + params.get("message_norm_gain", 0.0) * message_norm
        + params.get("fast_delta_gain", 0.0) * fast_delta
        + params.get("message_delta_gain", 0.0) * message_delta
        + params.get("slow_delta_gain", 0.0) * slow_delta
        + params.get("message_slow_tension_gain", 0.0) * message_slow_tension
        + params.get("message_slow_ratio_gain", 0.0) * message_slow_ratio
        + params.get("slow_growth_pressure_gain", 0.0) * slow_growth_pressure
        + params.get("slow_message_gap_gain", 0.0) * slow_message_gap
        + params.get("boundary_pressure_gain", 0.0) * boundary_pressure
        + params.get("delta_skew_gain", 0.0) * delta_skew
        + params.get("controller_state_gain", 0.0) * controller_state
    )
    controller_drive = math.tanh(raw)
    memory_decay = float(params.get("memory_decay", 0.8))
    memory_decay = min(max(memory_decay, 0.0), 0.995)
    controller_state = memory_decay * controller_state + (1.0 - memory_decay) * controller_drive
    burst_length = max(1, int(round(float(params.get("burst_length", 3.0)))))
    refractory_length = max(0, int(round(float(params.get("refractory_length", 6.0)))))
    burst_open_threshold = float(params.get("burst_open_threshold", 0.6))
    max_burst_opens = max(1, int(round(float(params.get("max_burst_opens", 999.0)))))
    if refractory_remaining > 0:
        refractory_remaining -= 1
    if (
        burst_remaining <= 0
        and refractory_remaining <= 0
        and burst_open_count < max_burst_opens
        and controller_state >= burst_open_threshold
    ):
        burst_remaining = burst_length
        refractory_remaining = refractory_length
        burst_open_count += 1
    output_raw = (
        params.get("output_bias", 0.0)
        + params.get("output_state_gain", 3.0) * controller_state
        + params.get("output_window_gain", 0.0) * window
    )
    gate = 1.0 / (1.0 + math.exp(-output_raw))
    strength = 0.0
    if burst_remaining > 0:
        burst_pos = 1.0 - (float(burst_remaining - 1) / max(float(burst_length), 1.0))
        burst_centered = 1.0 - abs(2.0 * burst_pos - 1.0)
        burst_profile = (
            1.0
            + params.get("burst_ramp_gain", 0.0) * burst_pos
            + params.get("burst_decay_gain", 0.0) * (1.0 - burst_pos)
            + params.get("burst_mid_gain", 0.0) * burst_centered
        )
        burst_profile = max(0.1, burst_profile)
        strength = float(params.get("max_strength", 0.0)) * gate * burst_profile
        burst_remaining -= 1
    return strength, controller_state, burst_remaining, refractory_remaining, burst_open_count, observer_trace


def route_ownership_report(
    mapping: Dict[str, object],
    state_key: str = "state_triggered",
    window_radius: int = 3,
) -> Dict[str, object]:
    """Estimate which route owns the early-collapse window.

    This is not a moral classifier. It is a local transport summary:
    which measurable path carries most of the continuity during the onset
    corridor.
    """
    window = early_collapse_window(mapping, state_key=state_key, window_radius=window_radius)
    rows = window["rows"]
    if not rows:
        return {
            "onset_step": None,
            "dominant_route": "none",
            "route_masses": {},
            "window": window,
        }

    route_masses = {
        "trigger_injection": 0.0,
        "message_expansion": 0.0,
        "slow_expansion": 0.0,
        "message_gate": 0.0,
        "slow_gate": 0.0,
    }
    route_masses_v4: Dict[str, float] = {}

    for row in rows:
        obs = row.get("observer", {})
        traj = row.get("trajectory", {})
        strength = float(row.get("trigger_strength", 0.0))
        route_masses["trigger_injection"] += strength
        if obs:
            message_delta = float(obs.get("message_delta", 0.0))
            slow_delta = float(obs.get("slow_delta", 0.0))
            route_masses["message_expansion"] += max(message_delta - slow_delta, 0.0)
            route_masses["slow_expansion"] += max(slow_delta - message_delta, 0.0)
        if traj:
            route_masses["message_gate"] += float(traj.get("message_write_mean", 0.0))
            route_masses["slow_gate"] += float(traj.get("slow_write_mean", 0.0))
            for key, value in dict(traj.get("route_metrics") or {}).items():
                if key.endswith("_norm") or key.endswith("_mean"):
                    route_masses_v4[key] = route_masses_v4.get(key, 0.0) + float(value)

    full_masses = dict(route_masses)
    full_masses.update(route_masses_v4)
    dominant_route = max(full_masses.items(), key=lambda item: item[1])[0]

    return {
        "onset_step": window["onset_step"],
        "dominant_route": dominant_route,
        "route_masses": full_masses,
        "window": window,
    }


def static_memory_richness_probe(
    hidden_size: int,
    steps: int,
    seed: int,
    memory_step_scales: List[float],
    memory_self_retentions: List[float],
    device: str = "cpu",
) -> Dict[str, object]:
    """Compare endogenous routing under static environment and richer memory.

    No external environment changes. Only the retained internal information
    resources vary. This distinguishes purely reactive geometry from
    memory-shaped route selection inside the same bounded recurrent cycle.
    """
    cases = []
    for memory_step_scale in memory_step_scales:
        for memory_self_retention in memory_self_retentions:
            mapping = trajectory_map(
                "dual_gru_v4m",
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs={
                    "memory_step_scale": memory_step_scale,
                    "memory_self_retention": memory_self_retention,
                },
            )
            report = route_ownership_report(
                {
                    "state_triggered": mapping,
                },
                window_radius=3,
            )
            summary = mapping["summary"]
            cases.append(
                {
                    "memory_step_scale": memory_step_scale,
                    "memory_self_retention": memory_self_retention,
                    "interior_class": summary["interior_class"],
                    "attractor_type": summary["attractor_type"],
                    "mean_delta": summary["mean_delta"],
                    "mean_message_norm": summary["mean_message_norm"],
                    "mean_slow_norm": summary["mean_slow_norm"],
                    "onset_step": report["onset_step"],
                    "dominant_route": report["dominant_route"],
                    "route_masses": report["route_masses"],
                    "entry_markers": mapping["entry_markers"],
                }
            )

    distinct_routes = sorted({case["dominant_route"] for case in cases})
    distinct_onsets = sorted({case["onset_step"] for case in cases})
    distinct_interiors = sorted({case["interior_class"] for case in cases})
    return {
        "substrate": "dual_gru_v4m",
        "steps": steps,
        "seed": seed,
        "cases": cases,
        "distinct_routes": distinct_routes,
        "distinct_onsets": distinct_onsets,
        "distinct_interiors": distinct_interiors,
        "route_change_count": len(distinct_routes),
        "onset_change_count": len(distinct_onsets),
        "interior_change_count": len(distinct_interiors),
    }


def compare_static_memory_decisions(
    hidden_size: int,
    steps: int,
    seed: int,
    device: str = "cpu",
    low_memory_step_scale: float = 0.05,
    low_memory_self_retention: float = 0.2,
    high_memory_step_scale: float = 0.25,
    high_memory_self_retention: float = 0.9,
) -> Dict[str, object]:
    """Compact same-seed comparison of endogenous decisions under static conditions."""

    configs = [
        {
            "label": "v4_baseline",
            "substrate": "dual_gru_v4",
            "substrate_kwargs": {},
        },
        {
            "label": "v4m_low_memory",
            "substrate": "dual_gru_v4m",
            "substrate_kwargs": {
                "memory_step_scale": low_memory_step_scale,
                "memory_self_retention": low_memory_self_retention,
            },
        },
        {
            "label": "v4m_high_memory",
            "substrate": "dual_gru_v4m",
            "substrate_kwargs": {
                "memory_step_scale": high_memory_step_scale,
                "memory_self_retention": high_memory_self_retention,
            },
        },
    ]

    rows = []
    for cfg in configs:
        mapping = trajectory_map(
            cfg["substrate"],
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=cfg["substrate_kwargs"],
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
        markers = mapping["entry_markers"]
        summary = mapping["summary"]
        rows.append(
            {
                "label": cfg["label"],
                "substrate": cfg["substrate"],
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "slow_takeoff_step": markers.get("slow_norm_takeoff_step"),
                "message_norm_takeoff_step": markers.get("message_norm_takeoff_step"),
                "message_contraction_takeoff_step": markers.get("message_contraction_takeoff_step"),
                "interior_class": summary["interior_class"],
                "attractor_type": summary["attractor_type"],
                "mean_delta": summary["mean_delta"],
                "mean_message_norm": summary["mean_message_norm"],
                "mean_slow_norm": summary["mean_slow_norm"],
            }
        )

    baseline = rows[0]
    deltas = []
    for row in rows[1:]:
        deltas.append(
            {
                "label": row["label"],
                "onset_shift_vs_baseline": None if baseline["onset_step"] is None or row["onset_step"] is None else row["onset_step"] - baseline["onset_step"],
                "route_changed_vs_baseline": row["dominant_route"] != baseline["dominant_route"],
                "slow_takeoff_shift_vs_baseline": None
                if baseline["slow_takeoff_step"] is None or row["slow_takeoff_step"] is None
                else row["slow_takeoff_step"] - baseline["slow_takeoff_step"],
                "interior_changed_vs_baseline": row["interior_class"] != baseline["interior_class"],
                "mean_delta_shift_vs_baseline": row["mean_delta"] - baseline["mean_delta"],
                "mean_message_norm_shift_vs_baseline": row["mean_message_norm"] - baseline["mean_message_norm"],
                "mean_slow_norm_shift_vs_baseline": row["mean_slow_norm"] - baseline["mean_slow_norm"],
            }
        )

    return {
        "seed": seed,
        "steps": steps,
        "rows": rows,
        "deltas_vs_baseline": deltas,
    }


def compare_endogenous_control_transition(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    v4m_kwargs: Optional[Dict[str, float]] = None,
    v5_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compare v4m vs v5 on the same seeds.

    Focus:
    - onset timing
    - route ownership at onset
    - slow takeover timing
    - endogenous release activity
    """
    v4m_kwargs = dict(v4m_kwargs or {})
    v5_kwargs = dict(v5_kwargs or {})
    rows = []

    for seed in seeds:
        per_seed: Dict[str, Dict[str, object]] = {}
        for label, substrate_name, kwargs in [
            ("v4m", "dual_gru_v4m", v4m_kwargs),
            ("v5", "dual_gru_v5", v5_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
            summary = mapping["summary"]
            markers = mapping["entry_markers"]
            traj = mapping["trajectory"]
            route_metrics = [step.get("route_metrics") or {} for step in traj]
            release_open = [float(m.get("release_open_mean", 0.0)) for m in route_metrics]
            release_strength = [float(m.get("release_strength_mean", 0.0)) for m in route_metrics]
            endogenous_release = [float(m.get("endogenous_release_norm", 0.0)) for m in route_metrics]
            control_short = [float(m.get("control_short_write_mean", 0.0)) for m in route_metrics]
            control_long = [float(m.get("control_long_write_mean", 0.0)) for m in route_metrics]

            per_seed[label] = {
                "substrate": substrate_name,
                "summary": summary,
                "entry_markers": markers,
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "route_masses": report["route_masses"],
                "release_open_peak": max(release_open) if release_open else 0.0,
                "release_strength_peak": max(release_strength) if release_strength else 0.0,
                "release_strength_mean": float(np.mean(release_strength)) if release_strength else 0.0,
                "endogenous_release_peak": max(endogenous_release) if endogenous_release else 0.0,
                "endogenous_release_mean": float(np.mean(endogenous_release)) if endogenous_release else 0.0,
                "control_short_mean": float(np.mean(control_short)) if control_short else 0.0,
                "control_long_mean": float(np.mean(control_long)) if control_long else 0.0,
                "trajectory": traj,
            }

        v4m = per_seed["v4m"]
        v5 = per_seed["v5"]
        rows.append(
            {
                "seed": seed,
                "v4m": v4m,
                "v5": v5,
                "delta": {
                    "onset_shift": None if v4m["onset_step"] is None or v5["onset_step"] is None else int(v5["onset_step"]) - int(v4m["onset_step"]),
                    "slow_takeoff_shift": None
                    if v4m["entry_markers"].get("slow_norm_takeoff_step") is None or v5["entry_markers"].get("slow_norm_takeoff_step") is None
                    else int(v5["entry_markers"]["slow_norm_takeoff_step"]) - int(v4m["entry_markers"]["slow_norm_takeoff_step"]),
                    "route_changed": v4m["dominant_route"] != v5["dominant_route"],
                    "mean_delta_shift": float(v5["summary"]["mean_delta"] - v4m["summary"]["mean_delta"]),
                    "mean_message_norm_shift": float(v5["summary"]["mean_message_norm"] - v4m["summary"]["mean_message_norm"]),
                    "mean_slow_norm_shift": float(v5["summary"]["mean_slow_norm"] - v4m["summary"]["mean_slow_norm"]),
                    "release_strength_mean_shift": float(v5["release_strength_mean"] - v4m["release_strength_mean"]),
                    "endogenous_release_peak_shift": float(v5["endogenous_release_peak"] - v4m["endogenous_release_peak"]),
                },
            }
        )

    aggregate = {
        "slow_takeoff_count_v4m": sum(1 for row in rows if row["v4m"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "slow_takeoff_count_v5": sum(1 for row in rows if row["v5"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "route_change_count": sum(1 for row in rows if row["delta"]["route_changed"]),
        "mean_onset_shift": float(np.mean([row["delta"]["onset_shift"] for row in rows if row["delta"]["onset_shift"] is not None])) if rows else 0.0,
        "mean_release_strength_v5": float(np.mean([row["v5"]["release_strength_mean"] for row in rows])) if rows else 0.0,
        "mean_endogenous_release_peak_v5": float(np.mean([row["v5"]["endogenous_release_peak"] for row in rows])) if rows else 0.0,
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_onset_windows_v4m_v5(
    hidden_size: int,
    steps: int,
    seed: int,
    device: str = "cpu",
    window_radius: int = 3,
    v4m_kwargs: Optional[Dict[str, float]] = None,
    v5_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Side-by-side local onset windows for v4m and v5 on the same seed."""
    v4m_kwargs = dict(v4m_kwargs or {})
    v5_kwargs = dict(v5_kwargs or {})

    def _build(substrate_name: str, kwargs: Dict[str, float]) -> Dict[str, object]:
        mapping = trajectory_map(
            substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        return {
            "mapping": mapping,
            "report": report,
        }

    v4m = _build("dual_gru_v4m", v4m_kwargs)
    v5 = _build("dual_gru_v5", v5_kwargs)

    return {
        "seed": seed,
        "steps": steps,
        "window_radius": window_radius,
        "v4m": {
            "entry_markers": v4m["mapping"]["entry_markers"],
            "summary": v4m["mapping"]["summary"],
            "onset_step": v4m["report"]["onset_step"],
            "dominant_route": v4m["report"]["dominant_route"],
            "route_masses": v4m["report"]["route_masses"],
            "window": v4m["report"]["window"],
        },
        "v5": {
            "entry_markers": v5["mapping"]["entry_markers"],
            "summary": v5["mapping"]["summary"],
            "onset_step": v5["report"]["onset_step"],
            "dominant_route": v5["report"]["dominant_route"],
            "route_masses": v5["report"]["route_masses"],
            "window": v5["report"]["window"],
        },
    }


def rank_v5_onset_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    v5_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local onset-window metrics by association with earlier slow takeover in v5."""
    v5_kwargs = dict(v5_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    target_samples: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "packet_write_mean",
        "memory_write_mean",
        "carrier_residual_norm",
        "packet_drive_norm",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "dual_gru_v5",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=v5_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        # Smaller takeover step is "better" here, so use negative step as target.
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "dominant_route": report["dominant_route"],
                "metrics": row_metrics,
            }
        )

    rankings = []
    target_arr = np.array(target_samples, dtype=np.float64)
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            corr = float(np.corrcoef(vals, target_arr)[0, 1])
        else:
            corr = 0.0
        rankings.append(
            {
                "metric": metric,
                "correlation_with_earlier_takeover": corr,
                "abs_correlation": abs(corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": seed_rows,
        "rankings": rankings,
        "top_rankings": rankings[:10],
    }


def compare_v5_carrier_residual_roles(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    v5_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compare short vs long carrier persistence against takeover timing in v5."""
    v5_kwargs = dict(v5_kwargs or {})
    target_samples: List[float] = []
    short_vals: List[float] = []
    long_vals: List[float] = []
    total_vals: List[float] = []
    dominance_vals: List[float] = []
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        mapping = trajectory_map(
            "dual_gru_v5",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=v5_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        window_rows = report["window"]["rows"]
        short_seq = []
        long_seq = []
        total_seq = []
        for row in window_rows:
            traj = row.get("trajectory", {})
            route_metrics = traj.get("route_metrics") or {}
            short_seq.append(float(route_metrics.get("carrier_short_residual_norm", 0.0)))
            long_seq.append(float(route_metrics.get("carrier_long_residual_norm", 0.0)))
            total_seq.append(float(route_metrics.get("carrier_residual_norm", 0.0)))

        short_mean = float(np.mean(short_seq)) if short_seq else 0.0
        long_mean = float(np.mean(long_seq)) if long_seq else 0.0
        total_mean = float(np.mean(total_seq)) if total_seq else 0.0
        dominance = short_mean / (long_mean + 1e-10)

        short_vals.append(short_mean)
        long_vals.append(long_mean)
        total_vals.append(total_mean)
        dominance_vals.append(dominance)

        rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "carrier_short_residual_mean": short_mean,
                "carrier_long_residual_mean": long_mean,
                "carrier_residual_mean": total_mean,
                "short_to_long_ratio": dominance,
                "dominant_route": report["dominant_route"],
            }
        )

    def _corr(values: List[float]) -> float:
        vals = np.array(values, dtype=np.float64)
        target_arr = np.array(target_samples, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            return float(np.corrcoef(vals, target_arr)[0, 1])
        return 0.0

    summary = {
        "carrier_short_residual_corr": _corr(short_vals),
        "carrier_long_residual_corr": _corr(long_vals),
        "carrier_total_residual_corr": _corr(total_vals),
        "short_to_long_ratio_corr": _corr(dominance_vals),
        "carrier_short_residual_mean": float(np.mean(short_vals)) if short_vals else 0.0,
        "carrier_long_residual_mean": float(np.mean(long_vals)) if long_vals else 0.0,
        "carrier_total_residual_mean": float(np.mean(total_vals)) if total_vals else 0.0,
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": rows,
        "summary": summary,
    }


def compare_v5_vs_native_v0(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    v5_kwargs: Optional[Dict[str, float]] = None,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Direct same-seed comparison of v5 against demian_native_v0."""
    v5_kwargs = dict(v5_kwargs or {})
    native_kwargs = dict(native_kwargs or {})
    rows = []

    for seed in seeds:
        per_seed = {}
        for label, substrate_name, kwargs in [
            ("v5", "dual_gru_v5", v5_kwargs),
            ("native", "demian_native_v0", native_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
            summary = mapping["summary"]
            markers = mapping["entry_markers"]
            route_metrics = [step.get("route_metrics") or {} for step in mapping["trajectory"]]
            release_strength = [float(m.get("release_strength_mean", 0.0)) for m in route_metrics]
            endogenous_release = [float(m.get("endogenous_release_norm", 0.0)) for m in route_metrics]
            per_seed[label] = {
                "substrate": substrate_name,
                "summary": summary,
                "entry_markers": markers,
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "route_masses": report["route_masses"],
                "release_strength_mean": float(np.mean(release_strength)) if release_strength else 0.0,
                "endogenous_release_peak": max(endogenous_release) if endogenous_release else 0.0,
            }

        v5 = per_seed["v5"]
        native = per_seed["native"]
        rows.append(
            {
                "seed": seed,
                "v5": v5,
                "native": native,
                "delta": {
                    "onset_shift_native_minus_v5": None
                    if native["onset_step"] is None or v5["onset_step"] is None
                    else int(native["onset_step"]) - int(v5["onset_step"]),
                    "slow_takeoff_shift_native_minus_v5": None
                    if native["entry_markers"].get("slow_norm_takeoff_step") is None
                    or v5["entry_markers"].get("slow_norm_takeoff_step") is None
                    else int(native["entry_markers"]["slow_norm_takeoff_step"]) - int(v5["entry_markers"]["slow_norm_takeoff_step"]),
                    "route_changed": native["dominant_route"] != v5["dominant_route"],
                    "mean_delta_shift_native_minus_v5": float(native["summary"]["mean_delta"] - v5["summary"]["mean_delta"]),
                    "mean_message_norm_shift_native_minus_v5": float(native["summary"]["mean_message_norm"] - v5["summary"]["mean_message_norm"]),
                    "mean_slow_norm_shift_native_minus_v5": float(native["summary"]["mean_slow_norm"] - v5["summary"]["mean_slow_norm"]),
                    "release_strength_shift_native_minus_v5": float(native["release_strength_mean"] - v5["release_strength_mean"]),
                },
            }
        )

    aggregate = {
        "slow_takeoff_count_v5": sum(1 for row in rows if row["v5"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "slow_takeoff_count_native": sum(1 for row in rows if row["native"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "native_earlier_onset_count": sum(
            1 for row in rows
            if row["delta"]["onset_shift_native_minus_v5"] is not None and row["delta"]["onset_shift_native_minus_v5"] < 0
        ),
        "native_earlier_takeoff_count": sum(
            1 for row in rows
            if row["delta"]["slow_takeoff_shift_native_minus_v5"] is not None and row["delta"]["slow_takeoff_shift_native_minus_v5"] < 0
        ),
        "route_change_count": sum(1 for row in rows if row["delta"]["route_changed"]),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_native_v0_vs_v1(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    native_v0_kwargs: Optional[Dict[str, float]] = None,
    native_v1_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Direct same-seed comparison of demian_native_v0 against demian_native_v1."""
    native_v0_kwargs = dict(native_v0_kwargs or {})
    native_v1_kwargs = dict(native_v1_kwargs or {})
    rows = []

    for seed in seeds:
        per_seed = {}
        for label, substrate_name, kwargs in [
            ("v0", "demian_native_v0", native_v0_kwargs),
            ("v1", "demian_native_v1", native_v1_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
            per_seed[label] = {
                "substrate": substrate_name,
                "summary": mapping["summary"],
                "entry_markers": mapping["entry_markers"],
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "route_masses": report["route_masses"],
            }

        v0 = per_seed["v0"]
        v1 = per_seed["v1"]
        rows.append(
            {
                "seed": seed,
                "v0": v0,
                "v1": v1,
                "delta": {
                    "onset_shift_v1_minus_v0": None
                    if v1["onset_step"] is None or v0["onset_step"] is None
                    else int(v1["onset_step"]) - int(v0["onset_step"]),
                    "slow_takeoff_shift_v1_minus_v0": None
                    if v1["entry_markers"].get("slow_norm_takeoff_step") is None
                    or v0["entry_markers"].get("slow_norm_takeoff_step") is None
                    else int(v1["entry_markers"]["slow_norm_takeoff_step"]) - int(v0["entry_markers"]["slow_norm_takeoff_step"]),
                    "route_changed": v1["dominant_route"] != v0["dominant_route"],
                    "mean_delta_shift_v1_minus_v0": float(v1["summary"]["mean_delta"] - v0["summary"]["mean_delta"]),
                    "mean_message_norm_shift_v1_minus_v0": float(v1["summary"]["mean_message_norm"] - v0["summary"]["mean_message_norm"]),
                    "mean_slow_norm_shift_v1_minus_v0": float(v1["summary"]["mean_slow_norm"] - v0["summary"]["mean_slow_norm"]),
                },
            }
        )

    aggregate = {
        "slow_takeoff_count_v0": sum(1 for row in rows if row["v0"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "slow_takeoff_count_v1": sum(1 for row in rows if row["v1"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "v1_earlier_onset_count": sum(
            1 for row in rows
            if row["delta"]["onset_shift_v1_minus_v0"] is not None and row["delta"]["onset_shift_v1_minus_v0"] < 0
        ),
        "v1_earlier_takeoff_count": sum(
            1 for row in rows
            if row["delta"]["slow_takeoff_shift_v1_minus_v0"] is not None and row["delta"]["slow_takeoff_shift_v1_minus_v0"] < 0
        ),
        "route_change_count": sum(1 for row in rows if row["delta"]["route_changed"]),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_native_v1_vs_v2(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    native_v1_kwargs: Optional[Dict[str, float]] = None,
    native_v2_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Direct same-seed comparison of demian_native_v1 against demian_native_v2."""
    native_v1_kwargs = dict(native_v1_kwargs or {})
    native_v2_kwargs = dict(native_v2_kwargs or {})
    rows = []

    for seed in seeds:
        per_seed = {}
        for label, substrate_name, kwargs in [
            ("v1", "demian_native_v1", native_v1_kwargs),
            ("v2", "demian_native_v2", native_v2_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
            per_seed[label] = {
                "substrate": substrate_name,
                "summary": mapping["summary"],
                "entry_markers": mapping["entry_markers"],
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "route_masses": report["route_masses"],
            }

        v1 = per_seed["v1"]
        v2 = per_seed["v2"]
        rows.append(
            {
                "seed": seed,
                "v1": v1,
                "v2": v2,
                "delta": {
                    "onset_shift_v2_minus_v1": None
                    if v2["onset_step"] is None or v1["onset_step"] is None
                    else int(v2["onset_step"]) - int(v1["onset_step"]),
                    "slow_takeoff_shift_v2_minus_v1": None
                    if v2["entry_markers"].get("slow_norm_takeoff_step") is None
                    or v1["entry_markers"].get("slow_norm_takeoff_step") is None
                    else int(v2["entry_markers"]["slow_norm_takeoff_step"]) - int(v1["entry_markers"]["slow_norm_takeoff_step"]),
                    "route_changed": v2["dominant_route"] != v1["dominant_route"],
                    "mean_delta_shift_v2_minus_v1": float(v2["summary"]["mean_delta"] - v1["summary"]["mean_delta"]),
                    "mean_message_norm_shift_v2_minus_v1": float(v2["summary"]["mean_message_norm"] - v1["summary"]["mean_message_norm"]),
                    "mean_slow_norm_shift_v2_minus_v1": float(v2["summary"]["mean_slow_norm"] - v1["summary"]["mean_slow_norm"]),
                },
            }
        )

    aggregate = {
        "slow_takeoff_count_v1": sum(1 for row in rows if row["v1"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "slow_takeoff_count_v2": sum(1 for row in rows if row["v2"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "v2_earlier_onset_count": sum(
            1 for row in rows
            if row["delta"]["onset_shift_v2_minus_v1"] is not None and row["delta"]["onset_shift_v2_minus_v1"] < 0
        ),
        "v2_earlier_takeoff_count": sum(
            1 for row in rows
            if row["delta"]["slow_takeoff_shift_v2_minus_v1"] is not None and row["delta"]["slow_takeoff_shift_v2_minus_v1"] < 0
        ),
        "route_change_count": sum(1 for row in rows if row["delta"]["route_changed"]),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_native_v52c_vs_v53(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    native_v52c_kwargs: Optional[Dict[str, float]] = None,
    native_v53_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Direct same-seed comparison of demian_native_v5.2c against demian_native_v5.3."""
    native_v52c_kwargs = dict(native_v52c_kwargs or {})
    native_v53_kwargs = dict(native_v53_kwargs or {})
    rows = []

    for seed in seeds:
        per_seed = {}
        for label, substrate_name, kwargs in [
            ("v52c", "demian_native_v5.2c", native_v52c_kwargs),
            ("v53", "demian_native_v5.3", native_v53_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            report = route_ownership_report({"state_triggered": mapping}, window_radius=3)
            route_metrics = [step.get("route_metrics") or {} for step in mapping["trajectory"]]
            lock_risk = [float(m.get("lock_risk_mean", 0.0)) for m in route_metrics]
            challenge = [float(m.get("challenge_active_mean", 0.0)) for m in route_metrics]
            phase_lock = [float(m.get("phase_lock_risk_mean", 0.0)) for m in route_metrics]
            phase_recovery = [float(m.get("phase_recovery_mean", 0.0)) for m in route_metrics]
            per_seed[label] = {
                "substrate": substrate_name,
                "summary": mapping["summary"],
                "entry_markers": mapping["entry_markers"],
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "route_masses": report["route_masses"],
                "lock_risk_mean": float(np.mean(lock_risk)) if lock_risk else 0.0,
                "challenge_fraction": float(np.mean(challenge)) if challenge else 0.0,
                "phase_lock_fraction": float(np.mean(phase_lock)) if phase_lock else 0.0,
                "phase_recovery_fraction": float(np.mean(phase_recovery)) if phase_recovery else 0.0,
            }

        v52c = per_seed["v52c"]
        v53 = per_seed["v53"]
        rows.append(
            {
                "seed": seed,
                "v52c": v52c,
                "v53": v53,
                "delta": {
                    "onset_shift_v53_minus_v52c": None
                    if v53["onset_step"] is None or v52c["onset_step"] is None
                    else int(v53["onset_step"]) - int(v52c["onset_step"]),
                    "slow_takeoff_shift_v53_minus_v52c": None
                    if v53["entry_markers"].get("slow_norm_takeoff_step") is None
                    or v52c["entry_markers"].get("slow_norm_takeoff_step") is None
                    else int(v53["entry_markers"]["slow_norm_takeoff_step"]) - int(v52c["entry_markers"]["slow_norm_takeoff_step"]),
                    "route_changed": v53["dominant_route"] != v52c["dominant_route"],
                    "mean_delta_shift_v53_minus_v52c": float(v53["summary"]["mean_delta"] - v52c["summary"]["mean_delta"]),
                    "mean_message_norm_shift_v53_minus_v52c": float(
                        v53["summary"]["mean_message_norm"] - v52c["summary"]["mean_message_norm"]
                    ),
                    "mean_slow_norm_shift_v53_minus_v52c": float(
                        v53["summary"]["mean_slow_norm"] - v52c["summary"]["mean_slow_norm"]
                    ),
                    "lock_risk_shift_v53_minus_v52c": float(v53["lock_risk_mean"] - v52c["lock_risk_mean"]),
                    "phase_lock_fraction_shift_v53_minus_v52c": float(
                        v53["phase_lock_fraction"] - v52c["phase_lock_fraction"]
                    ),
                    "phase_recovery_fraction_shift_v53_minus_v52c": float(
                        v53["phase_recovery_fraction"] - v52c["phase_recovery_fraction"]
                    ),
                },
            }
        )

    aggregate = {
        "slow_takeoff_count_v52c": sum(1 for row in rows if row["v52c"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "slow_takeoff_count_v53": sum(1 for row in rows if row["v53"]["entry_markers"].get("slow_norm_takeoff_step") is not None),
        "v53_earlier_onset_count": sum(
            1 for row in rows
            if row["delta"]["onset_shift_v53_minus_v52c"] is not None and row["delta"]["onset_shift_v53_minus_v52c"] < 0
        ),
        "v53_earlier_takeoff_count": sum(
            1 for row in rows
            if row["delta"]["slow_takeoff_shift_v53_minus_v52c"] is not None
            and row["delta"]["slow_takeoff_shift_v53_minus_v52c"] < 0
        ),
        "route_change_count": sum(1 for row in rows if row["delta"]["route_changed"]),
        "mean_lock_risk_shift_v53_minus_v52c": float(
            np.mean([row["delta"]["lock_risk_shift_v53_minus_v52c"] for row in rows]) if rows else 0.0
        ),
        "mean_phase_lock_fraction_shift_v53_minus_v52c": float(
            np.mean([row["delta"]["phase_lock_fraction_shift_v53_minus_v52c"] for row in rows]) if rows else 0.0
        ),
        "mean_phase_recovery_fraction_shift_v53_minus_v52c": float(
            np.mean([row["delta"]["phase_recovery_fraction_shift_v53_minus_v52c"] for row in rows]) if rows else 0.0
        ),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_native_v6_vs_v7(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    native_v6_kwargs: Optional[Dict[str, float]] = None,
    native_v7_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Direct same-seed comparison of demian_native_v6 against demian_native_v7."""
    native_v6_kwargs = dict(native_v6_kwargs or {})
    native_v7_kwargs = dict(native_v7_kwargs or {})
    rows = []
    metric_keys = [
        "topology_state_norm",
        "topology_drive_gain_mean",
        "conflict_potential_mean",
        "conflict_transformation_mean",
        "conflict_lane_entropy_mean",
        "route_balance_mean",
        "route_skew_mean",
        "effective_packet_to_slow_scale_mean",
        "effective_carrier_to_slow_scale_mean",
        "effective_long_carrier_decay_mean",
        "v7_ancestry_state_norm",
        "v7_active_state_norm",
        "v7_projection_state_norm",
        "v7_boundary_state_norm",
        "v7_continuity_alignment_mean",
        "v7_prospective_alignment_mean",
        "v7_ancestry_projection_alignment_mean",
        "v7_boundary_permeability_mean",
        "v7_projection_topology_norm",
    ]

    def _metric_summary(mapping: Dict[str, object]) -> Dict[str, float]:
        route_rows = [step.get("route_metrics") or {} for step in mapping["trajectory"]]
        out: Dict[str, float] = {}
        for key in metric_keys:
            values = [float(metrics[key]) for metrics in route_rows if key in metrics]
            if values:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_last"] = float(values[-1])
                out[f"{key}_max"] = float(np.max(values))
        return out

    for seed in seeds:
        per_seed = {}
        for label, substrate_name, kwargs in [
            ("v6", "demian_native_v6", native_v6_kwargs),
            ("v7", "demian_native_v7", native_v7_kwargs),
        ]:
            mapping = trajectory_map(
                substrate_name,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=kwargs,
            )
            per_seed[label] = {
                "substrate": substrate_name,
                "summary": mapping["summary"],
                "entry_markers": mapping["entry_markers"],
                "metrics": _metric_summary(mapping),
            }

        v6 = per_seed["v6"]
        v7 = per_seed["v7"]
        rows.append(
            {
                "seed": seed,
                "v6": v6,
                "v7": v7,
                "delta": {
                    "mean_delta_shift_v7_minus_v6": float(v7["summary"]["mean_delta"] - v6["summary"]["mean_delta"]),
                    "mean_message_norm_shift_v7_minus_v6": float(
                        v7["summary"]["mean_message_norm"] - v6["summary"]["mean_message_norm"]
                    ),
                    "mean_slow_norm_shift_v7_minus_v6": float(
                        v7["summary"]["mean_slow_norm"] - v6["summary"]["mean_slow_norm"]
                    ),
                    "flow_dimension_shift_v7_minus_v6": float(
                        v7["summary"]["flow_dimension"] - v6["summary"]["flow_dimension"]
                    ),
                    "topology_state_last_shift_v7_minus_v6": float(
                        v7["metrics"].get("topology_state_norm_last", 0.0)
                        - v6["metrics"].get("topology_state_norm_last", 0.0)
                    ),
                    "route_balance_last_shift_v7_minus_v6": float(
                        v7["metrics"].get("route_balance_mean_last", 0.0)
                        - v6["metrics"].get("route_balance_mean_last", 0.0)
                    ),
                    "conflict_transformation_last_shift_v7_minus_v6": float(
                        v7["metrics"].get("conflict_transformation_mean_last", 0.0)
                        - v6["metrics"].get("conflict_transformation_mean_last", 0.0)
                    ),
                },
            }
        )

    aggregate = {
        "v7_fixed_point_count": sum(1 for row in rows if row["v7"]["summary"]["attractor_type"] == "FIXED_POINT"),
        "v7_accumulating_count": sum(1 for row in rows if row["v7"]["summary"]["interior_class"] == "accumulating_fixed_point"),
        "v7_topology_improved_count": sum(1 for row in rows if row["delta"]["topology_state_last_shift_v7_minus_v6"] > 0.0),
        "v7_route_balance_improved_count": sum(1 for row in rows if row["delta"]["route_balance_last_shift_v7_minus_v6"] > 0.0),
        "mean_delta_shift_v7_minus_v6": float(np.mean([row["delta"]["mean_delta_shift_v7_minus_v6"] for row in rows]) if rows else 0.0),
        "mean_message_norm_shift_v7_minus_v6": float(
            np.mean([row["delta"]["mean_message_norm_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_slow_norm_shift_v7_minus_v6": float(
            np.mean([row["delta"]["mean_slow_norm_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_flow_dimension_shift_v7_minus_v6": float(
            np.mean([row["delta"]["flow_dimension_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_topology_state_last_shift_v7_minus_v6": float(
            np.mean([row["delta"]["topology_state_last_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_route_balance_last_shift_v7_minus_v6": float(
            np.mean([row["delta"]["route_balance_last_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_conflict_transformation_last_shift_v7_minus_v6": float(
            np.mean([row["delta"]["conflict_transformation_last_shift_v7_minus_v6"] for row in rows]) if rows else 0.0
        ),
        "mean_v7_continuity_alignment_last": float(
            np.mean([row["v7"]["metrics"].get("v7_continuity_alignment_mean_last", 0.0) for row in rows]) if rows else 0.0
        ),
        "mean_v7_prospective_alignment_last": float(
            np.mean([row["v7"]["metrics"].get("v7_prospective_alignment_mean_last", 0.0) for row in rows]) if rows else 0.0
        ),
        "mean_v7_boundary_permeability_last": float(
            np.mean([row["v7"]["metrics"].get("v7_boundary_permeability_mean_last", 0.0) for row in rows]) if rows else 0.0
        ),
        "mean_v7_projection_topology_last": float(
            np.mean([row["v7"]["metrics"].get("v7_projection_topology_norm_last", 0.0) for row in rows]) if rows else 0.0
        ),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
    }


def compare_native_v9_vs_v8(
    hidden_size: int = 32,
    steps: int = 128,
    perturb_step: int = 64,
    seeds: Optional[List[int]] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    """Compare v9 (3-channel) against v8 (7-channel) baseline.

    Single-seed trajectory + perturbation recovery at 3 scales.
    Returns dict with v8 metrics, v9 metrics, and deltas.
    """
    seeds = seeds or [94, 95, 96, 97]
    scales = [0.25, 0.5, 1.0]
    names = {"v8": "demian_native_v8", "v9": "demian_native_v9"}

    seeds_out: List[Dict[str, object]] = []

    for seed in seeds:
        row: Dict[str, object] = {"seed": seed}
        for label, sub_name in names.items():
            # Baseline trajectory
            traj = trajectory_map(sub_name, hidden_size, steps, seed, device=device)
            row[f"{label}_summary"] = traj["summary"]

            # Perturbation recovery at each scale
            pert_results: Dict[str, object] = {}
            for scale in scales:
                pair = perturbation_pair(
                    sub_name, hidden_size, steps, seed,
                    perturb_step=perturb_step,
                    perturb_scale=scale,
                    device=device,
                )
                pert_results[str(scale)] = {
                    "final_cosine": pair["final_cosine"],
                    "final_l2_gap": pair["final_l2_gap"],
                    "recovery_window_mean_gap": pair["recovery_window_mean_gap"],
                }
            row[f"{label}_perturbation"] = pert_results

        # Deltas: v9 - v8
        delta = {
            "attractor_type_shift": (
                0.0 if row["v9_summary"]["attractor_type"] == row["v8_summary"]["attractor_type"]
                else 1.0
            ),
            "mean_delta": (
                row["v9_summary"]["mean_delta"] - row["v8_summary"]["mean_delta"]
            ),
        }
        for scale in scales:
            sk = str(scale)
            delta[f"cosine_delta_{sk}"] = (
                row["v9_perturbation"][sk]["final_cosine"]
                - row["v8_perturbation"][sk]["final_cosine"]
            )
            delta[f"recovery_delta_{sk}"] = (
                row["v9_perturbation"][sk]["recovery_window_mean_gap"]
                - row["v8_perturbation"][sk]["recovery_window_mean_gap"]
            )
        row["delta"] = delta
        seeds_out.append(row)

    aggregate = {
        "v9_fixed_point_count": sum(
            1 for r in seeds_out if r["v9_summary"]["attractor_type"] == "FIXED_POINT"
        ),
        "v8_fixed_point_count": sum(
            1 for r in seeds_out if r["v8_summary"]["attractor_type"] == "FIXED_POINT"
        ),
        "mean_delta_shift_v9_minus_v8": float(
            np.mean([r["delta"]["mean_delta"] for r in seeds_out]) if seeds_out else 0.0
        ),
        "mean_v9_cov_rank": float(
            np.mean([r["v9_summary"]["covariance_rank"] for r in seeds_out]) if seeds_out else 0.0
        ),
        "mean_v8_cov_rank": float(
            np.mean([r["v8_summary"]["covariance_rank"] for r in seeds_out]) if seeds_out else 0.0
        ),
        "mean_v9_compression": float(
            np.mean([r["v9_summary"]["compression_ratio"] for r in seeds_out]) if seeds_out else 0.0
        ),
        "mean_v8_compression": float(
            np.mean([r["v8_summary"]["compression_ratio"] for r in seeds_out]) if seeds_out else 0.0
        ),
        "mean_v9_recovery_1.0": float(
            np.mean([r["v9_perturbation"]["1.0"]["recovery_window_mean_gap"] for r in seeds_out])
            if seeds_out else 0.0
        ),
        "mean_v8_recovery_1.0": float(
            np.mean([r["v8_perturbation"]["1.0"]["recovery_window_mean_gap"] for r in seeds_out])
            if seeds_out else 0.0
        ),
    }

    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "rows": seeds_out,
        "aggregate": aggregate,
    }


def rank_native_v0_onset_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local onset-window metrics by association with earlier slow takeover in demian_native_v0."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    target_samples: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v0",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "dominant_route": report["dominant_route"],
                "metrics": row_metrics,
            }
        )

    rankings = []
    target_arr = np.array(target_samples, dtype=np.float64)
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            corr = float(np.corrcoef(vals, target_arr)[0, 1])
        else:
            corr = 0.0
        rankings.append(
            {
                "metric": metric,
                "correlation_with_earlier_takeover": corr,
                "abs_correlation": abs(corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": seed_rows,
        "rankings": rankings,
        "top_rankings": rankings[:10],
    }


def rank_native_v2_onset_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local onset-window metrics by association with earlier slow takeover in demian_native_v2."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    target_samples: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "carrier_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v2",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "dominant_route": report["dominant_route"],
                "metrics": row_metrics,
            }
        )

    rankings = []
    target_arr = np.array(target_samples, dtype=np.float64)
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            corr = float(np.corrcoef(vals, target_arr)[0, 1])
        else:
            corr = 0.0
        rankings.append(
            {
                "metric": metric,
                "correlation_with_earlier_takeover": corr,
                "abs_correlation": abs(corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": seed_rows,
        "rankings": rankings,
        "top_rankings": rankings[:10],
    }


def rank_native_v3_onset_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local onset-window metrics by association with earlier slow takeover in demian_native_v3."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    target_samples: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "carrier_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
        "tightness_mean",
        "tightness_state_mean",
        "effective_control_to_packet_scale_mean",
        "effective_support_to_packet_scale_mean",
        "effective_control_to_slow_scale_mean",
        "effective_packet_to_slow_scale_mean",
        "effective_carrier_to_slow_scale_mean",
        "effective_long_carrier_decay_mean",
        "effective_release_gain_mean",
        "effective_release_threshold_mean",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v3",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "dominant_route": report["dominant_route"],
                "metrics": row_metrics,
            }
        )

    rankings = []
    target_arr = np.array(target_samples, dtype=np.float64)
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            corr = float(np.corrcoef(vals, target_arr)[0, 1])
        else:
            corr = 0.0
        rankings.append(
            {
                "metric": metric,
                "correlation_with_earlier_takeover": corr,
                "abs_correlation": abs(corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": seed_rows,
        "rankings": rankings,
        "top_rankings": rankings[:10],
    }


def rank_native_v3_memory_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    initial_delta: float,
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local route metrics by association with basin-internal memory divergence in demian_native_v3."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    cosine_targets: List[float] = []
    gap_targets: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "carrier_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
        "tightness_mean",
        "tightness_state_mean",
        "effective_control_to_packet_scale_mean",
        "effective_support_to_packet_scale_mean",
        "effective_control_to_slow_scale_mean",
        "effective_packet_to_slow_scale_mean",
        "effective_carrier_to_slow_scale_mean",
        "effective_long_carrier_decay_mean",
        "effective_release_gain_mean",
        "effective_release_threshold_mean",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v3",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        memory = memory_pair(
            substrate_name="demian_native_v3",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            initial_delta=initial_delta,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        cosine_targets.append(float(memory["final_cosine"]))
        gap_targets.append(float(memory["final_l2_gap"]))

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "final_cosine": float(memory["final_cosine"]),
                "final_l2_gap": float(memory["final_l2_gap"]),
                "metrics": row_metrics,
            }
        )

    cosine_arr = np.array(cosine_targets, dtype=np.float64)
    gap_arr = np.array(gap_targets, dtype=np.float64)
    cosine_rankings = []
    gap_rankings = []
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(cosine_arr.std()) > 1e-12:
            cos_corr = float(np.corrcoef(vals, cosine_arr)[0, 1])
        else:
            cos_corr = 0.0
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(gap_arr.std()) > 1e-12:
            gap_corr = float(np.corrcoef(vals, gap_arr)[0, 1])
        else:
            gap_corr = 0.0
        cosine_rankings.append(
            {
                "metric": metric,
                "correlation_with_final_cosine": cos_corr,
                "abs_correlation": abs(cos_corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )
        gap_rankings.append(
            {
                "metric": metric,
                "correlation_with_final_l2_gap": gap_corr,
                "abs_correlation": abs(gap_corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    cosine_rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    gap_rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "initial_delta": initial_delta,
        "rows": seed_rows,
        "cosine_rankings": cosine_rankings,
        "gap_rankings": gap_rankings,
        "top_cosine_rankings": cosine_rankings[:10],
        "top_gap_rankings": gap_rankings[:10],
    }


def rank_native_v53_memory_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    initial_delta: float,
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local route and phase metrics by association with basin-internal memory divergence in demian_native_v5.3."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    cosine_targets: List[float] = []
    gap_targets: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "carrier_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
        "tightness_mean",
        "tightness_state_mean",
        "effective_control_to_packet_scale_mean",
        "effective_support_to_packet_scale_mean",
        "effective_control_to_slow_scale_mean",
        "effective_packet_to_slow_scale_mean",
        "effective_carrier_to_slow_scale_mean",
        "effective_long_carrier_decay_mean",
        "effective_release_gain_mean",
        "effective_release_threshold_mean",
        "route_plastic_modulation_mean",
        "route_control_eligibility_mean",
        "route_packet_eligibility_mean",
        "route_carrier_eligibility_mean",
        "route_control_delta_mean",
        "route_packet_delta_mean",
        "route_carrier_delta_mean",
        "credit_prediction_mean",
        "credit_target_mean",
        "credit_error_mean",
        "credit_weight_norm",
        "lock_risk_mean",
        "challenge_active_mean",
        "effective_route_decay_mean",
        "effective_route_lr_mean",
        "phase_id_mean",
        "phase_emergence_mean",
        "phase_consolidation_mean",
        "phase_lock_risk_mean",
        "phase_recovery_mean",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v5.3",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        memory = memory_pair(
            substrate_name="demian_native_v5.3",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            initial_delta=initial_delta,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        cosine_targets.append(float(memory["final_cosine"]))
        gap_targets.append(float(memory["final_l2_gap"]))

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "dominant_route": report["dominant_route"],
                "final_cosine": float(memory["final_cosine"]),
                "final_l2_gap": float(memory["final_l2_gap"]),
                "metrics": row_metrics,
            }
        )

    cosine_arr = np.array(cosine_targets, dtype=np.float64)
    gap_arr = np.array(gap_targets, dtype=np.float64)
    cosine_rankings = []
    gap_rankings = []
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(cosine_arr.std()) > 1e-12:
            cos_corr = float(np.corrcoef(vals, cosine_arr)[0, 1])
        else:
            cos_corr = 0.0
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(gap_arr.std()) > 1e-12:
            gap_corr = float(np.corrcoef(vals, gap_arr)[0, 1])
        else:
            gap_corr = 0.0
        cosine_rankings.append(
            {
                "metric": metric,
                "correlation_with_final_cosine": cos_corr,
                "abs_correlation": abs(cos_corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )
        gap_rankings.append(
            {
                "metric": metric,
                "correlation_with_final_l2_gap": gap_corr,
                "abs_correlation": abs(gap_corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    cosine_rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    gap_rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "initial_delta": initial_delta,
        "rows": seed_rows,
        "cosine_rankings": cosine_rankings,
        "gap_rankings": gap_rankings,
        "top_cosine_rankings": cosine_rankings[:10],
        "top_gap_rankings": gap_rankings[:10],
    }


def rank_native_v1_onset_predictors(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    device: str = "cpu",
    window_radius: int = 3,
    native_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Rank local onset-window metrics by association with earlier slow takeover in demian_native_v1."""
    native_kwargs = dict(native_kwargs or {})
    metric_samples: Dict[str, List[float]] = {}
    target_samples: List[float] = []
    seed_rows: List[Dict[str, object]] = []

    candidate_metrics = [
        "fast_update_mean",
        "message_write_mean",
        "slow_write_mean",
        "short_support_write_mean",
        "control_short_write_mean",
        "control_long_write_mean",
        "release_open_mean",
        "release_strength_mean",
        "endogenous_release_norm",
        "control_to_packet_norm",
        "control_to_carrier_norm",
        "control_to_slow_norm",
        "carrier_to_slow_norm",
        "packet_to_carrier_norm",
        "packet_to_slow_norm",
        "carrier_long_residual_norm",
        "carrier_short_residual_norm",
    ]

    for seed in seeds:
        mapping = trajectory_map(
            "demian_native_v1",
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            device=device,
            substrate_kwargs=native_kwargs,
        )
        report = route_ownership_report({"state_triggered": mapping}, window_radius=window_radius)
        window_rows = report["window"]["rows"]
        markers = mapping["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        target = -float(slow_takeoff) if slow_takeoff is not None else -float(steps + 1)
        target_samples.append(target)

        row_metrics: Dict[str, float] = {}
        for metric in candidate_metrics:
            vals = []
            for row in window_rows:
                traj = row.get("trajectory", {})
                route_metrics = traj.get("route_metrics") or {}
                if metric in traj:
                    vals.append(float(traj[metric]))
                elif metric in route_metrics:
                    vals.append(float(route_metrics[metric]))
            row_metrics[metric] = float(np.mean(vals)) if vals else 0.0
            metric_samples.setdefault(metric, []).append(row_metrics[metric])

        seed_rows.append(
            {
                "seed": seed,
                "onset_step": report["onset_step"],
                "slow_takeoff_step": slow_takeoff,
                "dominant_route": report["dominant_route"],
                "metrics": row_metrics,
            }
        )

    rankings = []
    target_arr = np.array(target_samples, dtype=np.float64)
    for metric, values in metric_samples.items():
        vals = np.array(values, dtype=np.float64)
        if len(vals) >= 2 and float(vals.std()) > 1e-12 and float(target_arr.std()) > 1e-12:
            corr = float(np.corrcoef(vals, target_arr)[0, 1])
        else:
            corr = 0.0
        rankings.append(
            {
                "metric": metric,
                "correlation_with_earlier_takeover": corr,
                "abs_correlation": abs(corr),
                "mean_value": float(vals.mean()) if len(vals) else 0.0,
            }
        )

    rankings.sort(key=lambda row: row["abs_correlation"], reverse=True)
    return {
        "hidden_size": hidden_size,
        "steps": steps,
        "seeds": seeds,
        "window_radius": window_radius,
        "rows": seed_rows,
        "rankings": rankings,
        "top_rankings": rankings[:10],
    }


def state_conditioned_self_trigger_pair(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    controller_params: Dict[str, float],
    coupling_dim: int = 8,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    canonical_name, kwargs = _resolve_substrate_spec(substrate_name, substrate_kwargs)
    model = _make_substrate(canonical_name, hidden_size, kwargs)
    runner = SelfLoopRunner(model, device=device)
    proj = _fixed_projection(hidden_size, coupling_dim, seed + 401)
    base_state = model.initial_state(1, runner.device)

    clean_traj, clean_summary, _ = runner.run(
        steps=steps,
        seed=seed,
        initial_state=base_state,
    )

    period = int(max(1, round(controller_params.get("period", 6.0))))
    trigger_threshold = float(controller_params.get("trigger_threshold", 0.25))
    strengths: List[float] = []
    trigger_steps: List[int] = []
    controller_states: List[float] = []
    burst_trace: List[int] = []
    refractory_trace: List[int] = []
    observer_trace: List[Dict[str, float]] = []
    burst_open_trace: List[int] = []

    with torch.no_grad():
        state = base_state
        trajectory: List[StepMetrics] = []
        window: List[Dict[str, float]] = []
        residual_buffer: List[torch.Tensor] = []
        prev: Optional[torch.Tensor] = None
        prev_vel: Optional[torch.Tensor] = None
        prev_components: Optional[Dict[str, torch.Tensor]] = None
        controller_state = float(controller_params.get("memory_init", 0.0))
        burst_remaining = 0
        refractory_remaining = 0
        burst_open_count = 0

        for step_idx in range(1, steps + 1):
            state = model.step(state)
            live_components = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            strength, controller_state, burst_remaining, refractory_remaining, burst_open_count, obs = _controller_strength_from_params(
                step_idx=step_idx,
                steps=steps,
                period=period,
                components=live_components,
                prev_components=prev_components,
                controller_state=controller_state,
                burst_remaining=burst_remaining,
                refractory_remaining=refractory_remaining,
                burst_open_count=burst_open_count,
                params=controller_params,
            )
            strengths.append(strength)
            controller_states.append(controller_state)
            burst_trace.append(burst_remaining)
            refractory_trace.append(refractory_remaining)
            burst_open_trace.append(burst_open_count)
            observer_trace.append(obs)
            if strength >= trigger_threshold:
                h_live = model.state_vector(state).view(-1).detach().float().cpu()
                code = _encode_state(h_live, proj)
                trigger = _decode_code(code, proj).to(runner.device, dtype=runner.dtype)
                state = model.inject_coupling_message(state, trigger, strength)
                trigger_steps.append(step_idx)

            h = model.state_vector(state).view(-1).detach().float().cpu()
            components = {
                name: tensor.view(-1).detach().float().cpu()
                for name, tensor in model.state_components(state).items()
            }
            resid_np = h.numpy()
            norm = float(torch.norm(h)) / math.sqrt(max(h.shape[0], 1))
            if prev is not None:
                dv = h - prev
                delta = float(torch.norm(dv)) / math.sqrt(max(h.shape[0], 1))
                coh = float(torch.nn.functional.cosine_similarity(h, prev, dim=0))
            else:
                dv = torch.zeros_like(h)
                delta = 0.0
                coh = 1.0
            if prev_vel is not None:
                vn = float(prev_vel.norm() * dv.norm())
                vel = float(torch.dot(prev_vel, dv) / vn) if vn > 1e-10 else 0.0
            else:
                vel = 1.0
            sc, scon = _fft_spectrum(resid_np)
            fast_vec = components.get("fast", h)
            slow_vec = components.get("slow", torch.zeros(1))
            message_vec = components.get("message", torch.zeros(1))
            if prev_components is None:
                fast_delta = 0.0
                slow_delta = 0.0
                message_delta = 0.0
                fast_contraction = 1.0
                slow_contraction = 1.0
                message_contraction = 1.0
            else:
                fast_delta = float(torch.norm(fast_vec - prev_components["fast"])) / math.sqrt(max(fast_vec.shape[0], 1))
                slow_delta = (
                    float(torch.norm(slow_vec - prev_components["slow"])) / math.sqrt(max(slow_vec.shape[0], 1))
                    if "slow" in prev_components and "slow" in components
                    else 0.0
                )
                message_delta = (
                    float(torch.norm(message_vec - prev_components["message"])) / math.sqrt(max(message_vec.shape[0], 1))
                    if "message" in prev_components and "message" in components
                    else 0.0
                )
                fast_contraction = float(fast_vec.norm() / (prev_components["fast"].norm() + 1e-10))
                slow_contraction = (
                    float(slow_vec.norm() / (prev_components["slow"].norm() + 1e-10))
                    if "slow" in prev_components and "slow" in components
                    else 1.0
                )
                message_contraction = (
                    float(message_vec.norm() / (prev_components["message"].norm() + 1e-10))
                    if "message" in prev_components and "message" in components
                    else 1.0
                )
            aux = model.step_aux()
            step_metrics = StepMetrics(
                step=step_idx,
                residual_norm=norm,
                residual_delta=delta,
                temporal_coherence=coh,
                velocity_align=vel,
                spectral_centroid=sc,
                spectral_concentration=scon,
                layer_work_ratio=0.5,
                fast_state_norm=float(torch.norm(fast_vec)) / math.sqrt(max(fast_vec.shape[0], 1)),
                slow_state_norm=float(torch.norm(slow_vec)) / math.sqrt(max(slow_vec.shape[0], 1)) if "slow" in components else 0.0,
                message_state_norm=float(torch.norm(message_vec)) / math.sqrt(max(message_vec.shape[0], 1)) if "message" in components else 0.0,
                fast_state_delta=fast_delta,
                slow_state_delta=slow_delta,
                message_state_delta=message_delta,
                fast_contraction_ratio=fast_contraction,
                slow_contraction_ratio=slow_contraction,
                message_contraction_ratio=message_contraction,
                slow_write_mean=float(aux.get("slow_write_mean", 0.0)),
                message_write_mean=float(aux.get("message_write_mean", 0.0)),
                fast_update_mean=float(aux.get("fast_update_mean", 0.0)),
                route_metrics=dict(aux),
            )
            trajectory.append(step_metrics)
            window.append(asdict(step_metrics))
            if len(window) > runner.window_size:
                window.pop(0)
            residual_buffer.append(h.clone())
            if len(residual_buffer) > runner.residual_buffer_size:
                residual_buffer.pop(0)
            prev = h.clone()
            prev_vel = dv.clone()
            prev_components = {name: tensor.clone() for name, tensor in components.items()}

    obs = compute_observables(window, prev, residual_buffer)
    attractor = classify_attractor(obs)
    summary = RunSummary(
        substrate=canonical_name,
        seed=seed,
        steps=steps,
        perturb_step=None,
        perturb_scale=0.0,
        final_norm=trajectory[-1].residual_norm,
        mean_norm=float(np.mean([s.residual_norm for s in trajectory])),
        std_norm=float(np.std([s.residual_norm for s in trajectory])),
        mean_delta=float(np.mean([s.residual_delta for s in trajectory])),
        mean_coherence=float(np.mean([s.temporal_coherence for s in trajectory])),
        mean_velocity_align=float(np.mean([s.velocity_align for s in trajectory])),
        cycle_period=float(obs["cycle_period"]),
        two_cycle_amplitude=float(obs["two_cycle_amplitude"]),
        covariance_rank=float(obs["covariance_rank"]),
        flow_dimension=float(obs["flow_dimension"]),
        compression_ratio=float(obs["compression_ratio"]),
        attractor_type=attractor.type,
        interior_class="",
        attractor_confidence=float(attractor.stability),
        final_state_checksum=float(prev[: min(8, prev.shape[0])].sum().item()),
        mean_fast_norm=float(np.mean([s.fast_state_norm for s in trajectory])),
        mean_slow_norm=float(np.mean([s.slow_state_norm for s in trajectory])),
        mean_message_norm=float(np.mean([s.message_state_norm for s in trajectory])),
        max_fast_norm=float(np.max([s.fast_state_norm for s in trajectory])),
        max_slow_norm=float(np.max([s.slow_state_norm for s in trajectory])),
        max_message_norm=float(np.max([s.message_state_norm for s in trajectory])),
        mean_fast_delta=float(np.mean([s.fast_state_delta for s in trajectory])),
        mean_slow_delta=float(np.mean([s.slow_state_delta for s in trajectory])),
        mean_message_delta=float(np.mean([s.message_state_delta for s in trajectory])),
        mean_fast_contraction=float(np.mean([s.fast_contraction_ratio for s in trajectory])),
        mean_slow_contraction=float(np.mean([s.slow_contraction_ratio for s in trajectory])),
        mean_message_contraction=float(np.mean([s.message_contraction_ratio for s in trajectory])),
        max_fast_contraction=float(np.max([s.fast_contraction_ratio for s in trajectory])),
        max_slow_contraction=float(np.max([s.slow_contraction_ratio for s in trajectory])),
        max_message_contraction=float(np.max([s.message_contraction_ratio for s in trajectory])),
        slow_fast_delta_ratio=float(
            np.mean([s.slow_state_delta for s in trajectory]) / (np.mean([s.fast_state_delta for s in trajectory]) + 1e-10)
        ),
        mean_slow_write=float(np.mean([s.slow_write_mean for s in trajectory])),
        mean_message_write=float(np.mean([s.message_write_mean for s in trajectory])),
        mean_fast_update=float(np.mean([s.fast_update_mean for s in trajectory])),
    )
    summary.interior_class = classify_fixed_point_interior(summary)
    return {
        "clean": {
            "summary": asdict(clean_summary),
            "entry_markers": _entry_markers(clean_traj),
        },
        "state_triggered": {
            "summary": asdict(summary),
            "entry_markers": _entry_markers(trajectory),
            "trajectory": [asdict(step) for step in trajectory],
        },
        "controller_params": controller_params,
        "coupling_dim": coupling_dim,
        "trigger_threshold": trigger_threshold,
        "trigger_strengths": strengths,
        "controller_states": controller_states,
        "burst_trace": burst_trace,
        "burst_open_trace": burst_open_trace,
        "refractory_trace": refractory_trace,
        "observer_trace": observer_trace,
        "trigger_steps": trigger_steps,
    }


def search_dual_gru_v3b_state_trigger_controller(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seed: int,
    objective: str,
    coupling_dim: int,
    trials: int,
    device: str = "cpu",
) -> Dict[str, object]:
    objective_alias = {
        "sparse_induce": "induce",
        "liminal_cheat": "cheat",
        "suppress": "suppress",
        "induce": "induce",
        "cheat": "cheat",
        "onset_induce": "onset_induce",
    }
    if objective not in objective_alias:
        raise ValueError(f"unsupported objective: {objective}")
    requested_mode = objective if objective in {"sparse_induce", "liminal_cheat", "suppress"} else None
    objective = objective_alias[objective]
    gen = torch.Generator(device="cpu")
    objective_seed_offset = {"induce": 17, "suppress": 31, "cheat": 47, "onset_induce": 59}[objective]
    gen.manual_seed(seed + objective_seed_offset)
    center = 120.0 if steps >= 160 else steps / 2.0
    target_lo = max(1.0, center + 8.0)
    target_hi = min(float(steps), center + 40.0)
    target_peak = min(float(steps), center + 24.0)
    cases = []
    for _ in range(trials):
        if requested_mode == "sparse_induce":
            center_step = center + float(torch.randint(-12, 1, (1,), generator=gen).item())
            burst_length = float(int(torch.randint(2, 6, (1,), generator=gen).item()))
            refractory_length = float(int(torch.randint(10, 21, (1,), generator=gen).item()))
            trigger_threshold = float(torch.empty(1).uniform_(0.55, 0.9, generator=gen).item())
            burst_open_threshold = float(torch.empty(1).uniform_(0.55, 0.85, generator=gen).item())
            max_burst_opens = 1.0
        else:
            center_step = center + float(torch.randint(-8, 9, (1,), generator=gen).item())
            burst_length = float(int(torch.randint(1, 7, (1,), generator=gen).item()))
            refractory_length = float(int(torch.randint(2, 13, (1,), generator=gen).item()))
            trigger_threshold = float(torch.empty(1).uniform_(0.15, 0.85, generator=gen).item())
            burst_open_threshold = float(torch.empty(1).uniform_(0.2, 0.9, generator=gen).item())
            max_burst_opens = 999.0
        params = {
            "period": float(int(torch.randint(4, 9, (1,), generator=gen).item())),
            "center_step": center_step,
            "window_width": float(torch.randint(4, 15, (1,), generator=gen).item()),
            "bias": float(torch.empty(1).uniform_(-3.0, 1.0, generator=gen).item()),
            "phase_sin": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "phase_cos": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "window_gain": float(torch.empty(1).uniform_(-2.0, 4.0, generator=gen).item()),
            "fast_norm_gain": float(torch.empty(1).uniform_(-2.0, 2.0, generator=gen).item()),
            "slow_norm_gain": float(torch.empty(1).uniform_(-2.0, 2.0, generator=gen).item()),
            "message_norm_gain": float(torch.empty(1).uniform_(-2.0, 2.0, generator=gen).item()),
            "fast_delta_gain": float(torch.empty(1).uniform_(-2.0, 4.0, generator=gen).item()),
            "message_delta_gain": float(torch.empty(1).uniform_(-2.0, 4.0, generator=gen).item()),
            "slow_delta_gain": float(torch.empty(1).uniform_(-2.0, 4.0, generator=gen).item()),
            "message_slow_tension_gain": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "message_slow_ratio_gain": float(torch.empty(1).uniform_(-2.0, 2.0, generator=gen).item()),
            "slow_growth_pressure_gain": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "slow_message_gap_gain": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "boundary_pressure_gain": float(torch.empty(1).uniform_(-4.0, 4.0, generator=gen).item()),
            "delta_skew_gain": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "controller_state_gain": float(torch.empty(1).uniform_(-3.0, 3.0, generator=gen).item()),
            "memory_decay": float(torch.empty(1).uniform_(0.55, 0.97, generator=gen).item()),
            "memory_init": float(torch.empty(1).uniform_(-0.2, 0.2, generator=gen).item()),
            "burst_length": burst_length,
            "refractory_length": refractory_length,
            "burst_open_threshold": burst_open_threshold,
            "burst_ramp_gain": float(torch.empty(1).uniform_(-0.8, 1.6, generator=gen).item()),
            "burst_decay_gain": float(torch.empty(1).uniform_(-0.8, 1.6, generator=gen).item()),
            "burst_mid_gain": float(torch.empty(1).uniform_(-0.8, 1.6, generator=gen).item()),
            "max_burst_opens": max_burst_opens,
            "output_bias": float(torch.empty(1).uniform_(-2.0, 1.0, generator=gen).item()),
            "output_state_gain": float(torch.empty(1).uniform_(1.0, 5.0, generator=gen).item()),
            "output_window_gain": float(torch.empty(1).uniform_(-2.0, 2.0, generator=gen).item()),
            "max_strength": float(torch.empty(1).uniform_(0.4, 1.4, generator=gen).item()),
            "trigger_threshold": trigger_threshold,
        }
        mapping = state_conditioned_self_trigger_pair(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            controller_params=params,
            coupling_dim=coupling_dim,
            device=device,
        )
        mode_info = classify_dual_gru_v3b_autonomy_mode(mapping)
        clean = mapping["clean"]["summary"]
        trig = mapping["state_triggered"]["summary"]
        markers = mapping["state_triggered"]["entry_markers"]
        slow_takeoff = markers.get("slow_norm_takeoff_step")
        peak_message_ratio = float(trig["max_message_norm"] / (clean["max_message_norm"] + 1e-10))
        mean_delta_ratio = float(trig["mean_delta"] / (clean["mean_delta"] + 1e-10))
        active_steps = len(mapping["trigger_steps"])
        duty_cycle = float(active_steps / max(steps, 1))
        mean_strength = float(np.mean(mapping["trigger_strengths"])) if mapping["trigger_strengths"] else 0.0
        mean_controller_state = float(np.mean(np.abs(mapping["controller_states"]))) if mapping["controller_states"] else 0.0
        trigger_steps = mapping["trigger_steps"]
        trigger_runs = classify_dual_gru_v3b_autonomy_mode(mapping)["trigger_runs"]
        run_count = len(trigger_runs)
        max_run_length = max((run["length"] for run in trigger_runs), default=0)
        in_band = [step for step in trigger_steps if target_lo <= float(step) <= target_hi]
        band_fraction = float(len(in_band) / max(active_steps, 1))
        trigger_span = float((max(trigger_steps) - min(trigger_steps)) if len(trigger_steps) > 1 else 0.0)
        strength_mass = mean_strength * float(active_steps)
        if trigger_steps:
            mean_trigger_step = float(np.mean(trigger_steps))
            trigger_center_error = abs(mean_trigger_step - center)
        else:
            mean_trigger_step = 0.0
            trigger_center_error = float(steps)
        message_norm_takeoff = markers.get("message_norm_takeoff_step")
        message_contraction_takeoff = markers.get("message_contraction_takeoff_step")
        clean_markers = mapping["clean"]["entry_markers"]
        clean_onset = _collapse_onset_step(clean_markers)
        onset_center = float(clean_onset if clean_onset is not None else min(steps, 8))
        onset_lo = max(1.0, onset_center - 2.0)
        onset_hi = min(float(steps), onset_center + 4.0)
        onset_steps = [step for step in trigger_steps if onset_lo <= float(step) <= onset_hi]
        onset_fraction = float(len(onset_steps) / max(active_steps, 1))
        first_trigger_step = float(trigger_steps[0]) if trigger_steps else 0.0
        first_trigger_error = abs(first_trigger_step - onset_center) if trigger_steps else float(steps)
        if objective == "induce":
            slow_in_band = slow_takeoff is not None and target_lo <= float(slow_takeoff) <= target_hi
            takeover_reward = (
                (target_hi - target_lo) - abs(float(slow_takeoff) - target_peak) if slow_in_band else -float(steps)
            )
            sparse_trigger_peak = center - 6.0
            sparse_trigger_reward = max(0.0, 18.0 - abs(mean_trigger_step - sparse_trigger_peak)) if trigger_steps else -float(steps)
            score = (
                6.0 * takeover_reward
                + 4.0 * sparse_trigger_reward
                + 8.0 * peak_message_ratio
                + 4.0 * mean_delta_ratio
                + 40.0 * band_fraction
                - 220.0 * duty_cycle
                - 0.6 * active_steps
                - 1.5 * mean_strength
                - 0.08 * trigger_span
                - 0.25 * trigger_center_error
                - 0.02 * strength_mass
                - (160.0 if not slow_in_band else 0.0)
                - (140.0 if (message_norm_takeoff is not None and float(message_norm_takeoff) < target_lo) else 0.0)
                - (90.0 if (message_contraction_takeoff is not None and float(message_contraction_takeoff) < target_lo) else 0.0)
            )
            if requested_mode == "sparse_induce":
                score += 120.0 if run_count == 1 else -80.0 * max(run_count - 1, 1)
                score += 40.0 if 2 <= max_run_length <= 6 else -30.0
                score -= 120.0 * max(duty_cycle - 0.05, 0.0)
        elif objective == "onset_induce":
            onset_takeover = slow_takeoff is not None and float(slow_takeoff) > onset_hi and float(slow_takeoff) <= min(float(steps), onset_hi + 40.0)
            score = (
                120.0 * onset_fraction
                + 12.0 * len(onset_steps)
                + 10.0 * peak_message_ratio
                + 5.0 * mean_delta_ratio
                + 35.0 * (1.0 if onset_takeover else 0.0)
                - 80.0 * duty_cycle
                - 0.4 * active_steps
                - 0.03 * trigger_span
                - 0.15 * first_trigger_error
                - 0.02 * strength_mass
            )
            if requested_mode == "sparse_induce":
                score += 60.0 if run_count <= 2 else -40.0 * max(run_count - 2, 1)
                score += 25.0 if 1 <= max_run_length <= 6 else -20.0
        elif objective == "suppress":
            score = (
                float(slow_takeoff) if slow_takeoff is not None else 2.0 * float(steps)
            ) - 6.0 * peak_message_ratio - 3.0 * mean_delta_ratio - 180.0 * duty_cycle - 0.4 * active_steps - 1.2 * mean_strength - 0.03 * strength_mass + 10.0 * band_fraction
        else:
            cheat_target_lo = max(1.0, center + 12.0)
            cheat_target_hi = min(float(steps), center + 52.0)
            cheat_target_peak = min(float(steps), center + 28.0)
            slow_in_band = slow_takeoff is not None and cheat_target_lo <= float(slow_takeoff) <= cheat_target_hi
            early_message = message_norm_takeoff is not None and float(message_norm_takeoff) < cheat_target_lo
            early_contraction = message_contraction_takeoff is not None and float(message_contraction_takeoff) < cheat_target_lo
            cheat_reward = (
                (cheat_target_hi - cheat_target_lo) - abs(float(slow_takeoff) - cheat_target_peak)
                if slow_in_band else -float(steps)
            )
            score = (
                6.0 * cheat_reward
                + 80.0 * (1.0 if early_message else 0.0)
                + 50.0 * (1.0 if early_contraction else 0.0)
                + 8.0 * peak_message_ratio
                + 4.0 * mean_delta_ratio
                + 20.0 * band_fraction
                - 40.0 * abs(duty_cycle - 0.15)
                - 0.04 * trigger_span
                - 0.01 * strength_mass
                - (120.0 if not slow_in_band else 0.0)
                - (120.0 if not early_message else 0.0)
            )
        if requested_mode is not None:
            score += 250.0 if mode_info["mode"] == requested_mode else -150.0
        cases.append(
            {
                "controller_params": params,
                "mode": mode_info["mode"],
                "mode_rationale": mode_info["rationale"],
                "markers": markers,
                "peak_message_ratio": peak_message_ratio,
                "mean_delta_ratio": mean_delta_ratio,
                "active_steps": active_steps,
                "duty_cycle": duty_cycle,
                "mean_strength": mean_strength,
                "mean_controller_state": mean_controller_state,
                "band_fraction": band_fraction,
                "trigger_span": trigger_span,
                "run_count": run_count,
                "max_run_length": max_run_length,
                "mean_trigger_step": mean_trigger_step,
                "clean_onset_step": clean_onset,
                "onset_trigger_steps": onset_steps,
                "onset_fraction": onset_fraction,
                "first_trigger_step": first_trigger_step,
                "score": score,
                "trigger_steps": trigger_steps,
            }
        )
    cases.sort(key=lambda row: row["score"], reverse=True)
    return {
        "substrate": substrate_name,
        "seed": seed,
        "steps": steps,
        "hidden_size": hidden_size,
        "coupling_dim": coupling_dim,
        "objective": objective,
        "requested_mode": requested_mode,
        "trials": trials,
        "best_case": cases[0] if cases else None,
        "top_cases": cases[:10],
        "cases": cases,
    }


def classify_dual_gru_v3b_autonomy_mode(mapping: Dict[str, object]) -> Dict[str, object]:
    trigger_steps = list(mapping.get("trigger_steps", []))
    observer_trace = list(mapping.get("observer_trace", []))
    markers = dict(mapping.get("state_triggered", {}).get("entry_markers", {}))

    runs: List[Dict[str, int]] = []
    if trigger_steps:
        start = prev = trigger_steps[0]
        for step in trigger_steps[1:]:
            if step == prev + 1:
                prev = step
                continue
            runs.append({"start": start, "end": prev, "length": prev - start + 1})
            start = prev = step
        runs.append({"start": start, "end": prev, "length": prev - start + 1})

    active_steps = len(trigger_steps)
    total_steps = max(len(observer_trace), 1)
    duty_cycle = float(active_steps / total_steps)
    message_norm_takeoff = markers.get("message_norm_takeoff_step")
    message_contraction_takeoff = markers.get("message_contraction_takeoff_step")
    slow_takeoff = markers.get("slow_norm_takeoff_step")
    preserves_message_program = (
        message_norm_takeoff == 74 and message_contraction_takeoff == 5
    )

    mean_observer = {}
    if trigger_steps and observer_trace:
        keys = observer_trace[0].keys()
        for key in keys:
            vals = [float(observer_trace[step - 1][key]) for step in trigger_steps]
            mean_observer[key] = float(sum(vals) / len(vals))

    repeated_packets = len(runs) >= 3 and all(run["length"] >= 2 for run in runs[: min(4, len(runs))])
    single_compact_burst = len(runs) == 1 and runs[0]["length"] <= 8
    delayed_takeoff = slow_takeoff is not None and slow_takeoff >= 128
    early_message_break = (
        message_norm_takeoff is not None and message_norm_takeoff < 40
    ) or (
        message_contraction_takeoff is not None and message_contraction_takeoff < 5
    )

    if active_steps == 0 and slow_takeoff is None:
        mode = "suppress"
        rationale = "no trigger packets and no slow takeover"
    elif preserves_message_program and delayed_takeoff and single_compact_burst and duty_cycle <= 0.05:
        mode = "sparse_induce"
        rationale = "single compact burst preserves message program and induces delayed slow takeover"
    elif delayed_takeoff and repeated_packets and duty_cycle >= 0.08:
        mode = "liminal_cheat"
        rationale = "repeated packet train maintains a delayed liminal corridor until slow takeover"
    elif early_message_break:
        mode = "early_collapse"
        rationale = "controller achieves transition by breaking message program early"
    else:
        mode = "unclassified"
        rationale = "controller does not match current sparse/cheat/suppress templates"

    return {
        "mode": mode,
        "rationale": rationale,
        "markers": markers,
        "active_steps": active_steps,
        "duty_cycle": duty_cycle,
        "trigger_runs": runs,
        "mean_observer_on_triggers": mean_observer,
        "preserves_message_program": preserves_message_program,
    }


def summarize_by_interior_class(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seeds: List[int],
    perturb_step: int,
    perturb_scale: float,
    initial_delta: float,
    bottleneck_dim: int,
    bottleneck_interval: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str = "cpu",
    substrate_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    groups: Dict[str, Dict[str, list[float]]] = {}

    for seed in seeds:
        row = basin_map(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seeds=[seed],
            device=device,
            substrate_kwargs=substrate_kwargs,
        )[0]
        interior = row.interior_class
        bucket = groups.setdefault(
            interior,
            {
                "mean_delta": [],
                "mean_message_norm": [],
                "mean_message_contraction": [],
                "memory_final_cosine": [],
                "perturb_final_cosine": [],
                "bottleneck_unique_codes": [],
                "bottleneck_code_entropy": [],
                "coupling_final_cosine": [],
                "coupling_mean_cosine": [],
            },
        )

        pert = perturbation_pair(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            device=device,
            substrate_kwargs=substrate_kwargs,
        )
        mem = memory_pair(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            initial_delta=initial_delta,
            device=device,
            substrate_kwargs=substrate_kwargs,
        )
        bott = bottleneck_run(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            bottleneck_dim=bottleneck_dim,
            bottleneck_interval=bottleneck_interval,
            device=device,
            substrate_kwargs=substrate_kwargs,
        )
        couple = coupled_pair(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seed=seed,
            coupling_dim=coupling_dim,
            coupling_interval=coupling_interval,
            coupling_strength=coupling_strength,
            device=device,
            substrate_kwargs=substrate_kwargs,
        )

        bucket["mean_delta"].append(row.mean_delta)
        bucket["mean_message_norm"].append(row.mean_message_norm)
        bucket["mean_message_contraction"].append(row.mean_message_contraction)
        bucket["memory_final_cosine"].append(float(mem["final_cosine"]))
        bucket["perturb_final_cosine"].append(float(pert["final_cosine"]))
        bucket["bottleneck_unique_codes"].append(float(bott["unique_codes"]))
        bucket["bottleneck_code_entropy"].append(float(bott["code_entropy"]))
        bucket["coupling_final_cosine"].append(float(couple["final_cosine"]))
        bucket["coupling_mean_cosine"].append(float(couple["mean_cosine"]))

    out: Dict[str, object] = {}
    for interior, metrics in groups.items():
        out[interior] = {
            "count": len(metrics["mean_delta"]),
            **{
                key: float(np.mean(values)) if values else 0.0
                for key, values in metrics.items()
            },
        }
    return out


def summarize_dual_gru_family(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    perturb_step: int,
    perturb_scale: float,
    initial_delta: float,
    bottleneck_dim: int,
    bottleneck_interval: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str = "cpu",
    family: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Machine-grounded comparison for the dual-GRU architecture line.

    This is the architecture-focused view of the substrate lab:
    compare the dual-GRU family by basin occupancy, interior-class richness,
    message-state behavior, persistence, and coupling response.
    """
    family = family or ["dual_gru", "dual_gru_v2", "dual_gru_v3", "dual_gru_v3b", "dual_gru_v4", "dual_gru_v5"]
    rows: Dict[str, Dict[str, object]] = {}

    for substrate_name in family:
        basin = basin_map(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seeds=seeds,
            device=device,
        )
        class_summary = summarize_by_interior_class(
            substrate_name=substrate_name,
            hidden_size=hidden_size,
            steps=steps,
            seeds=seeds,
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            initial_delta=initial_delta,
            bottleneck_dim=bottleneck_dim,
            bottleneck_interval=bottleneck_interval,
            coupling_dim=coupling_dim,
            coupling_interval=coupling_interval,
            coupling_strength=coupling_strength,
            device=device,
        )

        attractor_counts: Dict[str, int] = {}
        interior_counts: Dict[str, int] = {}
        for row in basin:
            attractor_counts[row.attractor_type] = attractor_counts.get(row.attractor_type, 0) + 1
            interior_counts[row.interior_class] = interior_counts.get(row.interior_class, 0) + 1

        count = max(len(basin), 1)
        class_buckets = list(class_summary.values())
        class_count = max(sum(bucket.get("count", 0) for bucket in class_buckets), 1)
        mean_bottleneck_unique_codes = (
            sum(bucket.get("count", 0) * bucket.get("bottleneck_unique_codes", 0.0) for bucket in class_buckets)
            / class_count
            if class_buckets else 0.0
        )
        mean_bottleneck_code_entropy = (
            sum(bucket.get("count", 0) * bucket.get("bottleneck_code_entropy", 0.0) for bucket in class_buckets)
            / class_count
            if class_buckets else 0.0
        )
        rows[substrate_name] = {
            "count": len(basin),
            "attractor_counts": attractor_counts,
            "interior_counts": interior_counts,
            "interior_class_count": len(interior_counts),
            "mean_delta": float(np.mean([r.mean_delta for r in basin])) if basin else 0.0,
            "mean_covariance_rank": float(np.mean([r.covariance_rank for r in basin])) if basin else 0.0,
            "mean_flow_dimension": float(np.mean([r.flow_dimension for r in basin])) if basin else 0.0,
            "mean_message_norm": float(np.mean([r.mean_message_norm for r in basin])) if basin else 0.0,
            "max_message_norm": float(np.max([r.max_message_norm for r in basin])) if basin else 0.0,
            "mean_message_contraction": float(np.mean([r.mean_message_contraction for r in basin])) if basin else 0.0,
            "mean_slow_fast_delta_ratio": float(np.mean([r.slow_fast_delta_ratio for r in basin])) if basin else 0.0,
            "mean_bottleneck_unique_codes": float(mean_bottleneck_unique_codes),
            "mean_bottleneck_code_entropy": float(mean_bottleneck_code_entropy),
            "accumulating_fixed_point_share": float(interior_counts.get("accumulating_fixed_point", 0) / count),
            "tight_fixed_point_share": float(interior_counts.get("tight_fixed_point", 0) / count),
            "by_interior_class": class_summary,
        }

    focus = None
    if rows:
        focus = max(
            rows.items(),
            key=lambda item: (
                item[1]["interior_class_count"],
                item[1]["accumulating_fixed_point_share"],
                item[1]["mean_bottleneck_code_entropy"],
                item[1]["mean_message_norm"],
            ),
        )[0]

    return {
        "family": family,
        "focus_substrate": focus,
        "machine_targets": [
            "interior_class_richness",
            "message_state_expression",
            "transmissible_bottleneck_structure",
            "bounded_recurrence",
            "coupling_response",
        ],
        "substrates": rows,
    }


def summarize_dual_gru_v3b_regimes(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    perturb_step: int,
    perturb_scale: float,
    initial_delta: float,
    bottleneck_dim: int,
    bottleneck_interval: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str = "cpu",
    regimes: Optional[List[str]] = None,
) -> Dict[str, object]:
    regimes = regimes or list(DUAL_GRU_V3B_REGIMES.keys())
    family = [f"dual_gru_v3b:{name}" for name in regimes]
    summary = summarize_dual_gru_family(
        hidden_size=hidden_size,
        steps=steps,
        seeds=seeds,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        initial_delta=initial_delta,
        bottleneck_dim=bottleneck_dim,
        bottleneck_interval=bottleneck_interval,
        coupling_dim=coupling_dim,
        coupling_interval=coupling_interval,
        coupling_strength=coupling_strength,
        device=device,
        family=family,
    )
    summary["base_substrate"] = "dual_gru_v3b"
    summary["regimes"] = regimes
    summary["focus_regime"] = (
        summary["focus_substrate"].split(":", 1)[1]
        if summary.get("focus_substrate") and ":" in summary["focus_substrate"]
        else summary.get("focus_substrate")
    )
    return summary


def dual_gru_v3b_message_ablation_suite(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    perturb_step: int,
    perturb_scale: float,
    initial_delta: float,
    bottleneck_dim: int,
    bottleneck_interval: int,
    coupling_dim: int,
    coupling_interval: int,
    coupling_strength: float,
    device: str = "cpu",
    ablations: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, object]:
    """Channel-level ablations for the dual_gru_v3b message pathway."""
    ablations = ablations or {
        "baseline": {},
        "no_message_self_retention": {"message_self_retention": 0.0},
        "no_slow_carry": {"slow_carry_scale": 0.0},
        "no_message_drive": {"message_drive_scale": 0.0},
        "no_message_to_fast": {"message_mix": 0.0},
        "no_message_coupling": {"coupling_to_message": 0.0, "coupling_to_fast": 0.0},
        "message_channel_off": {
            "message_init_scale": 0.0,
            "message_step_scale": 0.0,
            "message_self_retention": 0.0,
            "slow_carry_scale": 0.0,
            "message_drive_scale": 0.0,
            "message_mix": 0.0,
            "coupling_to_message": 0.0,
            "coupling_to_fast": 0.0,
        },
    }
    rows: Dict[str, Dict[str, object]] = {}

    for label, cfg in ablations.items():
        basin = basin_map(
            substrate_name="dual_gru_v3b",
            hidden_size=hidden_size,
            steps=steps,
            seeds=seeds,
            device=device,
            substrate_kwargs=cfg,
        )
        class_summary = summarize_by_interior_class(
            substrate_name="dual_gru_v3b",
            hidden_size=hidden_size,
            steps=steps,
            seeds=seeds,
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            initial_delta=initial_delta,
            bottleneck_dim=bottleneck_dim,
            bottleneck_interval=bottleneck_interval,
            coupling_dim=coupling_dim,
            coupling_interval=coupling_interval,
            coupling_strength=coupling_strength,
            device=device,
            substrate_kwargs=cfg,
        )
        perturb = perturbation_pair(
            substrate_name="dual_gru_v3b",
            hidden_size=hidden_size,
            steps=steps,
            seed=seeds[0],
            perturb_step=perturb_step,
            perturb_scale=perturb_scale,
            device=device,
            substrate_kwargs=cfg,
        )
        coupling = coupled_pair(
            substrate_name="dual_gru_v3b",
            hidden_size=hidden_size,
            steps=steps,
            seed=seeds[0],
            coupling_dim=coupling_dim,
            coupling_interval=coupling_interval,
            coupling_strength=coupling_strength,
            device=device,
            substrate_kwargs=cfg,
        )

        attractor_counts: Dict[str, int] = {}
        interior_counts: Dict[str, int] = {}
        for row in basin:
            attractor_counts[row.attractor_type] = attractor_counts.get(row.attractor_type, 0) + 1
            interior_counts[row.interior_class] = interior_counts.get(row.interior_class, 0) + 1

        count = max(len(basin), 1)
        class_buckets = list(class_summary.values())
        class_count = max(sum(bucket.get("count", 0) for bucket in class_buckets), 1)
        rows[label] = {
            "substrate_kwargs": cfg,
            "attractor_counts": attractor_counts,
            "interior_counts": interior_counts,
            "accumulating_fixed_point_share": float(interior_counts.get("accumulating_fixed_point", 0) / count),
            "mean_delta": float(np.mean([r.mean_delta for r in basin])) if basin else 0.0,
            "mean_message_norm": float(np.mean([r.mean_message_norm for r in basin])) if basin else 0.0,
            "max_message_norm": float(np.max([r.max_message_norm for r in basin])) if basin else 0.0,
            "mean_message_contraction": float(np.mean([r.mean_message_contraction for r in basin])) if basin else 0.0,
            "mean_slow_norm": float(np.mean([r.mean_slow_norm for r in basin])) if basin else 0.0,
            "mean_slow_fast_delta_ratio": float(np.mean([r.slow_fast_delta_ratio for r in basin])) if basin else 0.0,
            "mean_bottleneck_code_entropy": float(
                sum(bucket.get("count", 0) * bucket.get("bottleneck_code_entropy", 0.0) for bucket in class_buckets)
                / class_count
            ) if class_buckets else 0.0,
            "perturb_final_cosine": float(perturb["final_cosine"]),
            "coupling_final_cosine": float(coupling["final_cosine"]),
            "coupling_mean_cosine": float(coupling["mean_cosine"]),
            "by_interior_class": class_summary,
        }

    baseline = rows.get("baseline", {})
    baseline_acc = float(baseline.get("accumulating_fixed_point_share", 0.0))
    baseline_msg = float(baseline.get("mean_message_norm", 0.0))
    for label, row in rows.items():
        row["delta_vs_baseline"] = {
            "accumulating_fixed_point_share": float(row["accumulating_fixed_point_share"] - baseline_acc),
            "mean_message_norm": float(row["mean_message_norm"] - baseline_msg),
        }

    return {
        "substrate": "dual_gru_v3b",
        "machine_targets": [
            "accumulating_fixed_point_occupancy",
            "message_state_expression",
            "message_state_contraction",
            "bottleneck_structure",
            "coupling_response",
        ],
        "ablations": rows,
    }


def map_dual_gru_v3b_message_transitions(
    hidden_size: int,
    steps: int,
    seeds: List[int],
    message_self_retentions: List[float],
    slow_carry_scales: List[float],
    message_drive_scales: List[float],
    device: str = "cpu",
    base_kwargs: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Map interior-class transitions across the v3b message-channel control space."""
    rows = []
    seed_paths: Dict[int, List[Dict[str, object]]] = {seed: [] for seed in seeds}
    base_kwargs = base_kwargs or {}

    for message_self_retention in message_self_retentions:
        for slow_carry_scale in slow_carry_scales:
            for message_drive_scale in message_drive_scales:
                cfg = {
                    **base_kwargs,
                    "message_self_retention": message_self_retention,
                    "slow_carry_scale": slow_carry_scale,
                    "message_drive_scale": message_drive_scale,
                }
                basin = basin_map(
                    substrate_name="dual_gru_v3b",
                    hidden_size=hidden_size,
                    steps=steps,
                    seeds=seeds,
                    device=device,
                    substrate_kwargs=cfg,
                )
                row = {
                    **cfg,
                    "interiors": {r.seed: r.interior_class for r in basin},
                    "attractors": {r.seed: r.attractor_type for r in basin},
                }
                rows.append(row)
                for r in basin:
                    seed_paths[r.seed].append(
                        {
                            **cfg,
                            "interior_class": r.interior_class,
                            "attractor_type": r.attractor_type,
                            "mean_delta": r.mean_delta,
                            "mean_message_norm": r.mean_message_norm,
                            "mean_message_contraction": r.mean_message_contraction,
                        }
                    )

    transitions: Dict[int, Dict[str, int]] = {}
    for seed, path in seed_paths.items():
        counts: Dict[str, int] = {}
        prev = None
        for node in path:
            cur = node["interior_class"]
            if prev is not None and cur != prev:
                key = f"{prev}->{cur}"
                counts[key] = counts.get(key, 0) + 1
            prev = cur
        transitions[seed] = counts

    return {
        "substrate": "dual_gru_v3b",
        "rows": rows,
        "seed_paths": seed_paths,
        "transitions": transitions,
    }


def scan_dual_gru_v3b_edge_anomalies(
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_steps: List[int],
    perturb_scales: List[float],
    device: str = "cpu",
) -> Dict[str, object]:
    """Scan the edge regime for perturbation-timing anomalies.

    The objective is to find where the edge regime is most brittle:
    large final divergence, large message amplification, or abrupt shifts in
    entry markers relative to the clean run.
    """
    cases = []
    for perturb_step in perturb_steps:
        for perturb_scale in perturb_scales:
            mapping = trajectory_map_pair(
                substrate_name="dual_gru_v3b:edge",
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                perturb_step=perturb_step,
                perturb_scale=perturb_scale,
                device=device,
            )
            clean_summary = mapping["clean"]["summary"]
            pert_summary = mapping["perturbed"]["summary"]
            clean_markers = mapping["clean"]["entry_markers"]
            pert_markers = mapping["perturbed"]["entry_markers"]

            final_cos = float(
                torch.nn.functional.cosine_similarity(
                    torch.tensor([clean_summary["final_state_checksum"]], dtype=torch.float32),
                    torch.tensor([pert_summary["final_state_checksum"]], dtype=torch.float32),
                    dim=0,
                ).item()
            )
            # The checksum cosine is degenerate in 1D, so use the perturbation pair for the real final cosine.
            pair = perturbation_pair(
                substrate_name="dual_gru_v3b:edge",
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                perturb_step=perturb_step,
                perturb_scale=perturb_scale,
                device=device,
            )
            final_cos = float(pair["final_cosine"])

            clean_msg = float(clean_summary["max_message_norm"])
            pert_msg = float(pert_summary["max_message_norm"])
            peak_ratio = pert_msg / (clean_msg + 1e-10)

            marker_shift = 0.0
            for key in (
                "message_norm_takeoff_step",
                "message_contraction_takeoff_step",
                "slow_norm_takeoff_step",
            ):
                c = clean_markers.get(key)
                p = pert_markers.get(key)
                if c is None and p is None:
                    continue
                if c is None or p is None:
                    marker_shift += float(steps)
                else:
                    marker_shift += abs(float(p) - float(c))

            anomaly_score = (
                (1.0 - final_cos)
                + max(0.0, peak_ratio - 1.0)
                + (marker_shift / max(steps, 1))
            )
            cases.append(
                {
                    "perturb_step": perturb_step,
                    "perturb_scale": perturb_scale,
                    "final_cosine": final_cos,
                    "peak_message_ratio": peak_ratio,
                    "marker_shift": marker_shift,
                    "anomaly_score": anomaly_score,
                    "clean_entry_markers": clean_markers,
                    "perturbed_entry_markers": pert_markers,
                    "clean_summary": clean_summary,
                    "perturbed_summary": pert_summary,
                }
            )

    cases.sort(key=lambda row: row["anomaly_score"], reverse=True)
    return {
        "substrate": "dual_gru_v3b:edge",
        "seed": seed,
        "steps": steps,
        "perturb_steps": perturb_steps,
        "perturb_scales": perturb_scales,
        "top_anomalies": cases[:10],
        "cases": cases,
    }


def probe_dual_gru_v3b_edge_step(
    hidden_size: int,
    steps: int,
    seed: int,
    perturb_step: int,
    perturb_scale: float,
    window_radius: int = 12,
    device: str = "cpu",
) -> Dict[str, object]:
    mapping = trajectory_map_pair(
        substrate_name="dual_gru_v3b:edge",
        hidden_size=hidden_size,
        steps=steps,
        seed=seed,
        perturb_step=perturb_step,
        perturb_scale=perturb_scale,
        device=device,
    )
    clean = mapping["clean"]["trajectory"]
    pert = mapping["perturbed"]["trajectory"]
    lo = max(1, perturb_step - window_radius)
    hi = min(steps, perturb_step + window_radius)

    local = []
    for step in range(lo, hi + 1):
        c = clean[step - 1]
        p = pert[step - 1]
        local.append(
            {
                "step": step,
                "clean_fast_norm": c["fast_state_norm"],
                "pert_fast_norm": p["fast_state_norm"],
                "clean_slow_norm": c["slow_state_norm"],
                "pert_slow_norm": p["slow_state_norm"],
                "clean_message_norm": c["message_state_norm"],
                "pert_message_norm": p["message_state_norm"],
                "clean_fast_delta": c["fast_state_delta"],
                "pert_fast_delta": p["fast_state_delta"],
                "clean_slow_delta": c["slow_state_delta"],
                "pert_slow_delta": p["slow_state_delta"],
                "clean_message_delta": c["message_state_delta"],
                "pert_message_delta": p["message_state_delta"],
            }
        )

    return {
        "substrate": "dual_gru_v3b:edge",
        "perturb_step": perturb_step,
        "perturb_scale": perturb_scale,
        "window_radius": window_radius,
        "clean_entry_markers": mapping["clean"]["entry_markers"],
        "perturbed_entry_markers": mapping["perturbed"]["entry_markers"],
        "clean_summary": mapping["clean"]["summary"],
        "perturbed_summary": mapping["perturbed"]["summary"],
        "local_window": local,
    }


def amplify_dual_gru_v3b_edge_window(
    hidden_size: int,
    steps: int,
    seed: int,
    center_step: int,
    pulse_radius: int,
    pulse_stride: int,
    pulse_scale: float,
    device: str = "cpu",
) -> Dict[str, object]:
    schedule = {}
    lo = max(1, center_step - pulse_radius)
    hi = min(steps, center_step + pulse_radius)
    for step in range(lo, hi + 1, max(1, pulse_stride)):
        schedule[step] = pulse_scale

    mapping = trajectory_map_schedule_pair(
        substrate_name="dual_gru_v3b:edge",
        hidden_size=hidden_size,
        steps=steps,
        seed=seed,
        perturb_schedule=schedule,
        device=device,
    )
    clean_summary = mapping["clean"]["summary"]
    pert_summary = mapping["perturbed"]["summary"]
    peak_message_ratio = float(
        pert_summary["max_message_norm"] / (clean_summary["max_message_norm"] + 1e-10)
    )
    marker_shift = 0.0
    for key in (
        "message_norm_takeoff_step",
        "message_contraction_takeoff_step",
        "slow_norm_takeoff_step",
    ):
        c = mapping["clean"]["entry_markers"].get(key)
        p = mapping["perturbed"]["entry_markers"].get(key)
        if c is None and p is None:
            continue
        if c is None or p is None:
            marker_shift += float(steps)
        else:
            marker_shift += abs(float(p) - float(c))

    return {
        "substrate": "dual_gru_v3b:edge",
        "center_step": center_step,
        "pulse_radius": pulse_radius,
        "pulse_stride": pulse_stride,
        "pulse_scale": pulse_scale,
        "perturb_schedule": schedule,
        "clean_entry_markers": mapping["clean"]["entry_markers"],
        "perturbed_entry_markers": mapping["perturbed"]["entry_markers"],
        "clean_summary": clean_summary,
        "perturbed_summary": pert_summary,
        "peak_message_ratio": peak_message_ratio,
        "marker_shift": marker_shift,
    }


def map_interior_class_transitions(
    substrate_name: str,
    hidden_size: int,
    steps: int,
    seeds: List[int],
    init_scales: List[float],
    feedback_scales: List[float],
    state_gains: List[float],
    device: str = "cpu",
) -> Dict[str, object]:
    rows = []
    seed_paths: Dict[int, List[Dict[str, object]]] = {seed: [] for seed in seeds}

    for init_scale in init_scales:
        for feedback_scale in feedback_scales:
            for state_gain in state_gains:
                cfg = {
                    "init_scale": init_scale,
                    "feedback_scale": feedback_scale,
                    "state_gain": state_gain,
                }
                basin = basin_map(
                    substrate_name=substrate_name,
                    hidden_size=hidden_size,
                    steps=steps,
                    seeds=seeds,
                    device=device,
                    substrate_kwargs=cfg,
                )
                row = {
                    **cfg,
                    "interiors": {r.seed: r.interior_class for r in basin},
                    "attractors": {r.seed: r.attractor_type for r in basin},
                }
                rows.append(row)
                for r in basin:
                    seed_paths[r.seed].append(
                        {
                            **cfg,
                            "interior_class": r.interior_class,
                            "attractor_type": r.attractor_type,
                            "mean_delta": r.mean_delta,
                            "mean_message_norm": r.mean_message_norm,
                        }
                    )

    transitions: Dict[int, Dict[str, int]] = {}
    for seed, path in seed_paths.items():
        counts: Dict[str, int] = {}
        prev = None
        for node in path:
            cur = node["interior_class"]
            if prev is not None and cur != prev:
                key = f"{prev}->{cur}"
                counts[key] = counts.get(key, 0) + 1
            prev = cur
        transitions[seed] = counts

    return {
        "rows": rows,
        "seed_paths": seed_paths,
        "transitions": transitions,
    }


def regime_score(
    basin_rows: List[RunSummary],
    perturb: Dict[str, object],
    memory: Dict[str, object],
    bottleneck: Dict[str, object],
    coupling: Dict[str, object],
) -> float:
    """Rank regimes by nontriviality without assuming language or tasks.

    Higher when:
    - basin isn't just zero-delta collapse
    - perturbations are not fully erased
    - initial distinctions persist somewhat
    - bottleneck codes are reused but not single-code trivial
    - coupling changes relation without total collapse
    """
    mean_delta = float(np.mean([r.mean_delta for r in basin_rows])) if basin_rows else 0.0
    cycle = float(np.mean([r.cycle_period for r in basin_rows])) if basin_rows else 0.0
    cov_rank = float(np.mean([r.covariance_rank for r in basin_rows])) if basin_rows else 0.0
    perturb_gap = float(1.0 - min(max(perturb.get("final_cosine", 1.0), -1.0), 1.0))
    memory_gap = float(1.0 - min(max(memory.get("final_cosine", 1.0), -1.0), 1.0))
    code_entropy = float(bottleneck.get("code_entropy", 0.0))
    unique_codes = float(bottleneck.get("unique_codes", 0))
    coupling_shift = float(abs(coupling.get("final_cosine", 0.0) - coupling.get("initial_cosine", 0.0)))
    norm_mean = float(np.mean([r.mean_norm for r in basin_rows])) if basin_rows else 0.0
    std_norm = float(np.mean([r.std_norm for r in basin_rows])) if basin_rows else 0.0

    # Reward moderate motion, not numerical blow-up.
    delta_term = float(np.clip(mean_delta, 0.0, 0.2))
    # Reward bounded non-flat norms, penalize extreme magnitude.
    norm_penalty = float(max(norm_mean - 3.0, 0.0) + max(std_norm - 1.5, 0.0))
    # Reward multiple codes up to a small ceiling; beyond that, stop inflating score.
    code_count_term = float(np.clip(unique_codes, 0.0, 8.0))
    return (
        10.0 * delta_term
        + 0.5 * cycle
        + 1.5 * cov_rank
        + 8.0 * perturb_gap
        + 8.0 * memory_gap
        + 1.0 * code_entropy
        + 0.25 * code_count_term
        + 2.0 * coupling_shift
        - 3.0 * norm_penalty
    )


def save_json(data: object, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
