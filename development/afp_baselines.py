#!/usr/bin/env python3
"""Matched baseline probe for accumulating fixed-point (AFP) candidates.

The experiment separates three objects:

1. native recurrent state: the architecture's own recurrent state;
2. native surface: the architecture's exposed recurrent surface;
3. common readout: the same low-dimensional observation protocol applied to
   every architecture.

An AFP candidate is not a mathematical fixed point of the full state. It is a
window in which the surface/readout changes very little while the full latent
state continues to move measurably.

This module is deliberately a measurement protocol, not a superiority
benchmark. It runs untrained deterministic recurrent systems under matched
temporal inputs and approximately matched trainable parameter budgets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

from development.demian_v1_gate_state import DemianV1GateState, V1State

ArchitectureName = Literal["rnn", "gru", "lstm", "demian_v1"]
ARCHITECTURES: tuple[ArchitectureName, ...] = ("rnn", "gru", "lstm", "demian_v1")


@dataclass(frozen=True)
class AFPConfig:
    """Deterministic protocol configuration."""

    demian_hidden_size: int = 32
    readout_dim: int = 8
    histories: int = 32
    drive_steps: int = 64
    shared_tail_steps: int = 16
    settle_steps: int = 64
    analysis_window: int = 16
    probe_steps: int = 16
    input_scale: float = 0.15
    probe_scale: float = 0.20
    model_seed: int = 94
    history_seed: int = 1904
    shared_tail_seed: int = 2904
    readout_seed: int = 3904
    probe_seed: int = 4904
    surface_rel_epsilon: float = 1e-3
    latent_rel_floor: float = 5e-4
    latent_surface_ratio_floor: float = 5.0
    max_baseline_hidden: int = 512


class RecurrentAdapter:
    """Minimal recurrent-state interface used by the protocol."""

    name: ArchitectureName
    hidden_size: int
    input_size: int

    def initial_state(self) -> object:
        raise NotImplementedError

    def reset_runtime(self, step_index: int = 0) -> None:
        del step_index

    def step(self, state: object, coupling: torch.Tensor) -> object:
        raise NotImplementedError

    def native_surface(self, state: object) -> torch.Tensor:
        raise NotImplementedError

    def latent_vector(self, state: object) -> torch.Tensor:
        raise NotImplementedError

    def parameter_count(self) -> int:
        raise NotImplementedError


class TorchCellAdapter(RecurrentAdapter):
    """Wrapper around a PyTorch RNN/GRU/LSTM cell."""

    def __init__(
        self,
        kind: Literal["rnn", "gru", "lstm"],
        *,
        input_size: int,
        hidden_size: int,
        seed: int,
    ) -> None:
        self.name = kind
        self.input_size = input_size
        self.hidden_size = hidden_size
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if kind == "rnn":
                self.cell: nn.Module = nn.RNNCell(input_size, hidden_size, nonlinearity="tanh")
            elif kind == "gru":
                self.cell = nn.GRUCell(input_size, hidden_size)
            elif kind == "lstm":
                self.cell = nn.LSTMCell(input_size, hidden_size)
            else:  # pragma: no cover - type narrowing guard
                raise ValueError(f"unknown baseline kind: {kind}")
        self.cell.eval()

    def initial_state(self) -> object:
        h = torch.zeros(1, self.hidden_size)
        if self.name == "lstm":
            return h, torch.zeros_like(h)
        return h

    def step(self, state: object, coupling: torch.Tensor) -> object:
        x = coupling.view(1, self.input_size)
        with torch.no_grad():
            if self.name == "lstm":
                h, c = state  # type: ignore[misc]
                return self.cell(x, (h, c))
            return self.cell(x, state)  # type: ignore[arg-type]

    def native_surface(self, state: object) -> torch.Tensor:
        if self.name == "lstm":
            h, _ = state  # type: ignore[misc]
            return h.view(-1)
        return state.view(-1)  # type: ignore[union-attr]

    def latent_vector(self, state: object) -> torch.Tensor:
        if self.name == "lstm":
            h, c = state  # type: ignore[misc]
            return torch.cat([h.view(-1), c.view(-1)])
        return state.view(-1)  # type: ignore[union-attr]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.cell.parameters())


class DemianAdapter(RecurrentAdapter):
    """Adapter for the current six-channel Demian v1 substrate."""

    name: ArchitectureName = "demian_v1"

    def __init__(self, *, hidden_size: int, seed: int) -> None:
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.model = DemianV1GateState(hidden_size=hidden_size)
        self.model.eval()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self._initial_state = _clone_state(self.model.initial_state(1, torch.device("cpu")))
        self.reset_runtime()

    def initial_state(self) -> V1State:
        self.reset_runtime()
        return _clone_state(self._initial_state)  # type: ignore[return-value]

    def reset_runtime(self, step_index: int = 0) -> None:
        self.model._step_index = int(step_index)
        self.model._frozen_gate = None

    def step(self, state: object, coupling: torch.Tensor) -> V1State:
        current = state  # type: ignore[assignment]
        vector = coupling.view(1, self.input_size)
        current = self.model.inject_coupling_message(current, vector, 1.0)
        with torch.no_grad():
            return self.model.step(current)

    def native_surface(self, state: object) -> torch.Tensor:
        return self.model.state_vector(state).view(-1).detach()  # type: ignore[arg-type]

    def latent_vector(self, state: object) -> torch.Tensor:
        channels = state  # type: ignore[assignment]
        return torch.cat([channel.view(-1) for channel in channels]).detach()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())


def _clone_state(state: object) -> object:
    if isinstance(state, torch.Tensor):
        return state.detach().clone()
    if isinstance(state, tuple):
        return tuple(_clone_state(part) for part in state)
    raise TypeError(f"unsupported recurrent state type: {type(state)!r}")


def _rms(vector: torch.Tensor) -> float:
    flat = vector.view(-1).float()
    return float(torch.linalg.vector_norm(flat).item() / math.sqrt(max(flat.numel(), 1)))


def _relative_velocity(mean_delta: float, mean_norm: float) -> float:
    return mean_delta / max(mean_norm, 1e-12)


def _parameter_count_for_baseline(
    kind: Literal["rnn", "gru", "lstm"],
    *,
    input_size: int,
    hidden_size: int,
    seed: int,
) -> int:
    return TorchCellAdapter(
        kind,
        input_size=input_size,
        hidden_size=hidden_size,
        seed=seed,
    ).parameter_count()


def match_baseline_hidden_size(
    kind: Literal["rnn", "gru", "lstm"],
    *,
    target_parameters: int,
    input_size: int,
    seed: int,
    max_hidden: int,
) -> tuple[int, int]:
    """Find the integer hidden size with parameter count closest to Demian."""

    if max_hidden < 2:
        raise ValueError("max_hidden must be >= 2")
    best: tuple[int, int, int] | None = None
    for hidden_size in range(2, max_hidden + 1):
        count = _parameter_count_for_baseline(
            kind,
            input_size=input_size,
            hidden_size=hidden_size,
            seed=seed,
        )
        error = abs(count - target_parameters)
        candidate = (error, hidden_size, count)
        if best is None or candidate < best:
            best = candidate
        if count > target_parameters and best is not None and hidden_size > best[1] + 4:
            break
    assert best is not None
    _, hidden_size, count = best
    return hidden_size, count


def _orthogonal_readout(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    if output_dim < 1 or output_dim > input_dim:
        raise ValueError(f"readout_dim must be in [1, {input_dim}], got {output_dim}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(input_dim, output_dim, generator=generator)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q.T.contiguous()


def _project(surface: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return projection @ surface.view(-1)


def _build_histories(config: AFPConfig) -> list[torch.Tensor]:
    if config.shared_tail_steps < 0 or config.shared_tail_steps > config.drive_steps:
        raise ValueError("shared_tail_steps must be between 0 and drive_steps")
    prefix_steps = config.drive_steps - config.shared_tail_steps

    prefix_generator = torch.Generator(device="cpu")
    prefix_generator.manual_seed(config.history_seed)
    tail_generator = torch.Generator(device="cpu")
    tail_generator.manual_seed(config.shared_tail_seed)

    if config.shared_tail_steps:
        shared_tail = config.input_scale * torch.tanh(
            torch.randn(
                config.shared_tail_steps,
                config.demian_hidden_size,
                generator=tail_generator,
            )
        )
    else:
        shared_tail = torch.empty(0, config.demian_hidden_size)

    histories: list[torch.Tensor] = []
    for _ in range(config.histories):
        prefix = config.input_scale * torch.tanh(
            torch.randn(
                prefix_steps,
                config.demian_hidden_size,
                generator=prefix_generator,
            )
        )
        histories.append(torch.cat([prefix, shared_tail.clone()], dim=0))
    return histories


def _probe_sequence(config: AFPConfig) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.probe_seed)
    sequence = torch.zeros(config.probe_steps, config.demian_hidden_size)
    if config.probe_steps:
        sequence[0] = config.probe_scale * torch.tanh(
            torch.randn(config.demian_hidden_size, generator=generator)
        )
    return sequence


def _tail_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    tail = values[-min(window, len(values)) :]
    return sum(tail) / len(tail)


def _history_metrics(
    adapter: RecurrentAdapter,
    *,
    state: object,
    projection: torch.Tensor,
    settle_steps: int,
    analysis_window: int,
) -> tuple[dict[str, Any], object, torch.Tensor, torch.Tensor]:
    zero = torch.zeros(adapter.input_size)

    previous_surface = adapter.native_surface(state).detach().clone()
    previous_readout = _project(previous_surface, projection).detach().clone()
    previous_latent = adapter.latent_vector(state).detach().clone()

    surface_deltas: list[float] = []
    readout_deltas: list[float] = []
    latent_deltas: list[float] = []
    surface_norms: list[float] = []
    readout_norms: list[float] = []
    latent_norms: list[float] = []

    for _ in range(settle_steps):
        state = adapter.step(state, zero)
        surface = adapter.native_surface(state).detach().clone()
        readout = _project(surface, projection).detach().clone()
        latent = adapter.latent_vector(state).detach().clone()

        surface_deltas.append(_rms(surface - previous_surface))
        readout_deltas.append(_rms(readout - previous_readout))
        latent_deltas.append(_rms(latent - previous_latent))
        surface_norms.append(_rms(surface))
        readout_norms.append(_rms(readout))
        latent_norms.append(_rms(latent))

        previous_surface = surface
        previous_readout = readout
        previous_latent = latent

    mean_surface_delta = _tail_mean(surface_deltas, analysis_window)
    mean_readout_delta = _tail_mean(readout_deltas, analysis_window)
    mean_latent_delta = _tail_mean(latent_deltas, analysis_window)
    mean_surface_norm = _tail_mean(surface_norms, analysis_window)
    mean_readout_norm = _tail_mean(readout_norms, analysis_window)
    mean_latent_norm = _tail_mean(latent_norms, analysis_window)

    native_surface_rel = _relative_velocity(mean_surface_delta, mean_surface_norm)
    common_readout_rel = _relative_velocity(mean_readout_delta, mean_readout_norm)
    latent_rel = _relative_velocity(mean_latent_delta, mean_latent_norm)

    metrics = {
        "native_surface_mean_delta": mean_surface_delta,
        "common_readout_mean_delta": mean_readout_delta,
        "latent_mean_delta": mean_latent_delta,
        "native_surface_mean_norm": mean_surface_norm,
        "common_readout_mean_norm": mean_readout_norm,
        "latent_mean_norm": mean_latent_norm,
        "native_surface_rel_velocity": native_surface_rel,
        "common_readout_rel_velocity": common_readout_rel,
        "latent_rel_velocity": latent_rel,
        "native_latent_velocity_ratio": latent_rel / max(native_surface_rel, 1e-12),
        "readout_latent_velocity_ratio": latent_rel / max(common_readout_rel, 1e-12),
        "native_surface_path_length": sum(surface_deltas[-min(analysis_window, len(surface_deltas)) :]),
        "common_readout_path_length": sum(readout_deltas[-min(analysis_window, len(readout_deltas)) :]),
        "latent_path_length": sum(latent_deltas[-min(analysis_window, len(latent_deltas)) :]),
    }
    return metrics, state, previous_readout, previous_latent


def _is_afp_candidate(
    *,
    surface_rel_velocity: float,
    latent_rel_velocity: float,
    config: AFPConfig,
) -> bool:
    return (
        surface_rel_velocity <= config.surface_rel_epsilon
        and latent_rel_velocity >= config.latent_rel_floor
        and latent_rel_velocity / max(surface_rel_velocity, 1e-12)
        >= config.latent_surface_ratio_floor
    )


def _select_pairs(finals: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for left in range(len(finals)):
        for right in range(left + 1, len(finals)):
            surface_gap = _rms(finals[left]["readout"] - finals[right]["readout"])
            latent_gap = _rms(finals[left]["latent"] - finals[right]["latent"])
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "surface_gap": surface_gap,
                    "latent_gap": latent_gap,
                    "latent_surface_gap_ratio": latent_gap / max(surface_gap, 1e-12),
                }
            )
    if not pairs:
        return {}

    by_surface = sorted(pairs, key=lambda item: item["surface_gap"])
    nearest = by_surface[0]
    collision_pool = by_surface[: max(5, math.ceil(len(by_surface) * 0.10))]
    collision = max(
        collision_pool,
        key=lambda item: (item["latent_surface_gap_ratio"], item["latent_gap"]),
    )
    return {
        "nearest_surface_pair": nearest,
        "selected_hidden_collision_pair": collision,
    }


def _future_gap(
    adapter: RecurrentAdapter,
    *,
    left_state: object,
    right_state: object,
    projection: torch.Tensor,
    sequence: torch.Tensor,
    start_step_index: int,
) -> dict[str, Any]:
    left = _clone_state(left_state)
    right = _clone_state(right_state)
    gaps: list[float] = []

    for branch, initial in (("left", left), ("right", right)):
        del branch, initial

    left_surfaces: list[torch.Tensor] = []
    right_surfaces: list[torch.Tensor] = []

    adapter.reset_runtime(start_step_index)
    for coupling in sequence:
        left = adapter.step(left, coupling)
        left_surfaces.append(_project(adapter.native_surface(left), projection).detach().clone())

    adapter.reset_runtime(start_step_index)
    for coupling in sequence:
        right = adapter.step(right, coupling)
        right_surfaces.append(_project(adapter.native_surface(right), projection).detach().clone())

    for left_surface, right_surface in zip(left_surfaces, right_surfaces, strict=True):
        gaps.append(_rms(left_surface - right_surface))

    return {
        "mean_gap": sum(gaps) / len(gaps) if gaps else 0.0,
        "max_gap": max(gaps) if gaps else 0.0,
        "final_gap": gaps[-1] if gaps else 0.0,
        "gaps": gaps,
    }


def _make_adapters(config: AFPConfig) -> tuple[dict[ArchitectureName, RecurrentAdapter], dict[str, Any]]:
    demian = DemianAdapter(hidden_size=config.demian_hidden_size, seed=config.model_seed)
    target_parameters = demian.parameter_count()

    adapters: dict[ArchitectureName, RecurrentAdapter] = {"demian_v1": demian}
    budget: dict[str, Any] = {
        "target_architecture": "demian_v1",
        "target_parameters": target_parameters,
        "models": {
            "demian_v1": {
                "hidden_size": config.demian_hidden_size,
                "parameters": target_parameters,
                "relative_error": 0.0,
            }
        },
    }

    for kind in ("rnn", "gru", "lstm"):
        hidden_size, parameter_count = match_baseline_hidden_size(
            kind,
            target_parameters=target_parameters,
            input_size=config.demian_hidden_size,
            seed=config.model_seed,
            max_hidden=config.max_baseline_hidden,
        )
        adapter = TorchCellAdapter(
            kind,
            input_size=config.demian_hidden_size,
            hidden_size=hidden_size,
            seed=config.model_seed,
        )
        adapters[kind] = adapter
        budget["models"][kind] = {
            "hidden_size": hidden_size,
            "parameters": parameter_count,
            "relative_error": abs(parameter_count - target_parameters) / max(target_parameters, 1),
        }

    return adapters, budget


def run_experiment(config: AFPConfig) -> dict[str, Any]:
    """Run the matched AFP baseline protocol and return a JSON-safe result."""

    if config.histories < 2:
        raise ValueError("histories must be >= 2")
    if config.settle_steps < 1:
        raise ValueError("settle_steps must be >= 1")
    if config.analysis_window < 1 or config.analysis_window > config.settle_steps:
        raise ValueError("analysis_window must be in [1, settle_steps]")

    histories = _build_histories(config)
    probe = _probe_sequence(config)
    adapters, budget = _make_adapters(config)

    results: dict[str, Any] = {
        "protocol": "afp-baseline-v1",
        "definition": {
            "native_afp": "native surface approximately stationary while full recurrent state continues moving",
            "projected_afp": "common low-dimensional readout approximately stationary while full recurrent state continues moving",
            "fixed_point_boundary": "AFP is a projected/observable regime, not a claim that the complete recurrent state is at a mathematical fixed point",
        },
        "config": asdict(config),
        "parameter_budget": budget,
        "architectures": {},
    }

    for architecture_index, name in enumerate(ARCHITECTURES):
        adapter = adapters[name]
        surface_dim = int(adapter.native_surface(adapter.initial_state()).numel())
        projection = _orthogonal_readout(
            surface_dim,
            config.readout_dim,
            config.readout_seed + architecture_index,
        )
        finals: list[dict[str, Any]] = []
        history_rows: list[dict[str, Any]] = []

        for history_index, sequence in enumerate(histories):
            state = adapter.initial_state()
            adapter.reset_runtime(0)
            for coupling in sequence:
                state = adapter.step(state, coupling)

            metrics, final_state, final_readout, final_latent = _history_metrics(
                adapter,
                state=state,
                projection=projection,
                settle_steps=config.settle_steps,
                analysis_window=config.analysis_window,
            )
            metrics["history_index"] = history_index
            metrics["native_afp_candidate"] = _is_afp_candidate(
                surface_rel_velocity=metrics["native_surface_rel_velocity"],
                latent_rel_velocity=metrics["latent_rel_velocity"],
                config=config,
            )
            metrics["projected_afp_candidate"] = _is_afp_candidate(
                surface_rel_velocity=metrics["common_readout_rel_velocity"],
                latent_rel_velocity=metrics["latent_rel_velocity"],
                config=config,
            )
            history_rows.append(metrics)
            finals.append(
                {
                    "state": _clone_state(final_state),
                    "readout": final_readout.detach().clone(),
                    "latent": final_latent.detach().clone(),
                }
            )

        pair_info = _select_pairs(finals)
        continuation: dict[str, Any] = {}
        selected = pair_info.get("selected_hidden_collision_pair")
        if selected:
            left_index = int(selected["left"])
            right_index = int(selected["right"])
            start_step = config.drive_steps + config.settle_steps
            zero_sequence = torch.zeros(config.probe_steps, config.demian_hidden_size)
            zero_future = _future_gap(
                adapter,
                left_state=finals[left_index]["state"],
                right_state=finals[right_index]["state"],
                projection=projection,
                sequence=zero_sequence,
                start_step_index=start_step,
            )
            probe_future = _future_gap(
                adapter,
                left_state=finals[left_index]["state"],
                right_state=finals[right_index]["state"],
                projection=projection,
                sequence=probe,
                start_step_index=start_step,
            )
            continuation = {
                "pre_future_surface_gap": selected["surface_gap"],
                "pre_future_latent_gap": selected["latent_gap"],
                "zero_future": zero_future,
                "shared_probe_future": probe_future,
                "probe_minus_zero_mean_gap": probe_future["mean_gap"] - zero_future["mean_gap"],
            }

        native_candidates = sum(bool(row["native_afp_candidate"]) for row in history_rows)
        projected_candidates = sum(bool(row["projected_afp_candidate"]) for row in history_rows)
        results["architectures"][name] = {
            "hidden_size": adapter.hidden_size,
            "native_surface_dim": surface_dim,
            "latent_dim": int(adapter.latent_vector(adapter.initial_state()).numel()),
            "parameter_count": adapter.parameter_count(),
            "histories": history_rows,
            "aggregate": {
                "native_afp_candidates": native_candidates,
                "projected_afp_candidates": projected_candidates,
                "native_afp_fraction": native_candidates / len(history_rows),
                "projected_afp_fraction": projected_candidates / len(history_rows),
                "mean_native_surface_rel_velocity": sum(
                    row["native_surface_rel_velocity"] for row in history_rows
                )
                / len(history_rows),
                "mean_common_readout_rel_velocity": sum(
                    row["common_readout_rel_velocity"] for row in history_rows
                )
                / len(history_rows),
                "mean_latent_rel_velocity": sum(row["latent_rel_velocity"] for row in history_rows)
                / len(history_rows),
            },
            "pair_search": pair_info,
            "continuation": continuation,
        }

    return results


def write_results(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "afp_baseline_results.json"
    csv_path = output_dir / "afp_baseline_summary.csv"

    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "architecture",
                "hidden_size",
                "parameter_count",
                "native_surface_dim",
                "latent_dim",
                "native_afp_candidates",
                "projected_afp_candidates",
                "native_afp_fraction",
                "projected_afp_fraction",
                "mean_native_surface_rel_velocity",
                "mean_common_readout_rel_velocity",
                "mean_latent_rel_velocity",
            ],
        )
        writer.writeheader()
        for architecture, payload in result["architectures"].items():
            aggregate = payload["aggregate"]
            writer.writerow(
                {
                    "architecture": architecture,
                    "hidden_size": payload["hidden_size"],
                    "parameter_count": payload["parameter_count"],
                    "native_surface_dim": payload["native_surface_dim"],
                    "latent_dim": payload["latent_dim"],
                    **aggregate,
                }
            )
    return json_path, csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/afp_baselines"))
    parser.add_argument("--demian-hidden-size", type=int, default=32)
    parser.add_argument("--readout-dim", type=int, default=8)
    parser.add_argument("--histories", type=int, default=32)
    parser.add_argument("--drive-steps", type=int, default=64)
    parser.add_argument("--shared-tail-steps", type=int, default=16)
    parser.add_argument("--settle-steps", type=int, default=64)
    parser.add_argument("--analysis-window", type=int, default=16)
    parser.add_argument("--probe-steps", type=int, default=16)
    parser.add_argument("--model-seed", type=int, default=94)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = AFPConfig(
        demian_hidden_size=args.demian_hidden_size,
        readout_dim=args.readout_dim,
        histories=args.histories,
        drive_steps=args.drive_steps,
        shared_tail_steps=args.shared_tail_steps,
        settle_steps=args.settle_steps,
        analysis_window=args.analysis_window,
        probe_steps=args.probe_steps,
        model_seed=args.model_seed,
    )
    result = run_experiment(config)
    json_path, csv_path = write_results(result, args.output_dir)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "parameter_budget": result["parameter_budget"],
        "aggregate": {
            name: payload["aggregate"]
            for name, payload in result["architectures"].items()
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
