"""Runtime helpers for stepping substrate instances.

This module is the active runner surface for new substrate work. The large
legacy substrate module still defines historical experiments, but scripts that
only need execution metrics should import this file through
``development.substrates.legacy`` or directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch

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


class SelfLoopRunner:
    def __init__(
        self,
        substrate: Any,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        window_size: int = 20,
        residual_buffer_size: int = 16,
        substrate_registry: Mapping[str, type] | None = None,
    ):
        self.substrate = substrate.to(device=device, dtype=dtype)
        self.device = torch.device(device)
        self.dtype = dtype
        self.window_size = window_size
        self.residual_buffer_size = residual_buffer_size
        self.substrate_registry = dict(substrate_registry or {})

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
                    state = self._apply_perturbation(
                        state,
                        perturb_schedule[step_idx],
                        perturb_mode,
                    )
                elif perturb_step is not None and step_idx == perturb_step:
                    state = self._apply_perturbation(
                        state,
                        perturb_scale,
                        perturb_mode,
                    )

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
                    fast_delta = float(
                        torch.norm(fast_vec - prev_components["fast"])
                    ) / math.sqrt(max(fast_vec.shape[0], 1))
                    slow_delta = (
                        float(torch.norm(slow_vec - prev_components["slow"]))
                        / math.sqrt(max(slow_vec.shape[0], 1))
                        if "slow" in prev_components and "slow" in components
                        else 0.0
                    )
                    message_delta = (
                        float(torch.norm(message_vec - prev_components["message"]))
                        / math.sqrt(max(message_vec.shape[0], 1))
                        if "message" in prev_components and "message" in components
                        else 0.0
                    )
                    fast_contraction = float(
                        fast_vec.norm() / (prev_components["fast"].norm() + 1e-10)
                    )
                    slow_contraction = (
                        float(
                            slow_vec.norm()
                            / (prev_components["slow"].norm() + 1e-10)
                        )
                        if "slow" in prev_components and "slow" in components
                        else 1.0
                    )
                    message_contraction = (
                        float(
                            message_vec.norm()
                            / (prev_components["message"].norm() + 1e-10)
                        )
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
                    fast_state_norm=float(torch.norm(fast_vec))
                    / math.sqrt(max(fast_vec.shape[0], 1)),
                    slow_state_norm=float(torch.norm(slow_vec))
                    / math.sqrt(max(slow_vec.shape[0], 1))
                    if "slow" in components
                    else 0.0,
                    message_state_norm=float(torch.norm(message_vec))
                    / math.sqrt(max(message_vec.shape[0], 1))
                    if "message" in components
                    else 0.0,
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
                prev_components = {
                    name: tensor.clone() for name, tensor in components.items()
                }

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
                mean_coherence=float(
                    np.mean([s.temporal_coherence for s in trajectory])
                ),
                mean_velocity_align=float(
                    np.mean([s.velocity_align for s in trajectory])
                ),
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
                mean_message_norm=float(
                    np.mean([s.message_state_norm for s in trajectory])
                ),
                max_fast_norm=float(np.max([s.fast_state_norm for s in trajectory])),
                max_slow_norm=float(np.max([s.slow_state_norm for s in trajectory])),
                max_message_norm=float(
                    np.max([s.message_state_norm for s in trajectory])
                ),
                mean_fast_delta=float(
                    np.mean([s.fast_state_delta for s in trajectory])
                ),
                mean_slow_delta=float(
                    np.mean([s.slow_state_delta for s in trajectory])
                ),
                mean_message_delta=float(
                    np.mean([s.message_state_delta for s in trajectory])
                ),
                mean_fast_contraction=float(
                    np.mean([s.fast_contraction_ratio for s in trajectory])
                ),
                mean_slow_contraction=float(
                    np.mean([s.slow_contraction_ratio for s in trajectory])
                ),
                mean_message_contraction=float(
                    np.mean([s.message_contraction_ratio for s in trajectory])
                ),
                max_fast_contraction=float(
                    np.max([s.fast_contraction_ratio for s in trajectory])
                ),
                max_slow_contraction=float(
                    np.max([s.slow_contraction_ratio for s in trajectory])
                ),
                max_message_contraction=float(
                    np.max([s.message_contraction_ratio for s in trajectory])
                ),
                slow_fast_delta_ratio=float(
                    np.mean([s.slow_state_delta for s in trajectory])
                    / (np.mean([s.fast_state_delta for s in trajectory]) + 1e-10)
                ),
                mean_slow_write=float(np.mean([s.slow_write_mean for s in trajectory])),
                mean_message_write=float(
                    np.mean([s.message_write_mean for s in trajectory])
                ),
                mean_fast_update=float(np.mean([s.fast_update_mean for s in trajectory])),
            )
            summary.interior_class = classify_fixed_point_interior(summary)
            return trajectory, summary, prev

    def _apply_perturbation(self, state: object, scale: float, mode: str = "noise") -> object:
        if scale <= 0:
            return state
        if mode in {"rss_negation", "surface_negation"}:
            surface = self.substrate.state_vector(state).to(
                device=self.device,
                dtype=self.dtype,
            )
            negated = (1.0 - scale) * surface - scale * surface
            return self.substrate.write_surface_state(state, negated)
        if mode != "noise":
            raise ValueError(f"Unknown perturbation mode: {mode}")
        if isinstance(state, tuple):
            return tuple(s + scale * torch.randn_like(s) for s in state)
        return state + scale * torch.randn_like(state)

    def _substrate_name(self) -> str:
        for name, cls in self.substrate_registry.items():
            if type(self.substrate) is cls:
                return name
        for name, cls in self.substrate_registry.items():
            if isinstance(self.substrate, cls):
                return name
        return self.substrate.__class__.__name__.lower()
