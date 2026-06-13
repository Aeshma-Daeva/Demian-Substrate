"""3D trajectory export for the current v9/v8 substrate workbench."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from development.substrates.legacy import compare_native_v9_vs_v8, trajectory_map

CURRENT_TARGET = "demian_native_v9"
CURRENT_COMPARISON = "demian_native_v8"
DEFAULT_SUBSTRATES = (CURRENT_TARGET, CURRENT_COMPARISON)
DEFAULT_SEEDS = (94, 95, 96, 97)
DEFAULT_PERTURB_SCALES = (0.25, 0.5, 1.0)
DEFAULT_ROUTE_METRICS = (
    "fast_update_mean",
    "slow_write_mean",
    "message_write_mean",
    "release_open_mean",
    "release_strength_mean",
    "release_pressure_mean",
    "release_drive_mean",
    "release_bias_norm",
    "endogenous_release_norm",
    "control_short_write_mean",
    "control_long_write_mean",
    "carrier_residual_norm",
    "lock_risk_mean",
    "challenge_active_mean",
    "phase_lock_risk_mean",
    "phase_recovery_mean",
)

CORE_METRICS = (
    "residual_norm",
    "residual_delta",
    "temporal_coherence",
    "velocity_align",
    "spectral_centroid",
    "spectral_concentration",
    "fast_state_norm",
    "slow_state_norm",
    "message_state_norm",
    "fast_state_delta",
    "slow_state_delta",
    "message_state_delta",
    "fast_contraction_ratio",
    "slow_contraction_ratio",
    "message_contraction_ratio",
)


def export_current_trajectory_3d(
    out_dir: Path | str = Path("data/substrate_lab"),
    run_name: str = "v9_v8_trajectory_3d_20260508",
    hidden_size: int = 32,
    steps: int = 128,
    perturb_step: int = 64,
    seeds: list[int] | None = None,
    perturb_scales: list[float] | None = None,
    device: str = "cpu",
    route_metric_keys: tuple[str, ...] = DEFAULT_ROUTE_METRICS,
    substrate_kwargs_by_name: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Export deterministic 3D trajectory data for v9 and v8.

    The projection uses PCA over scalar machine observables from
    ``trajectory_map`` output. It deliberately avoids raw hidden tensors so the
    exported artifact stays compact and interpretation remains about measured
    state dynamics rather than an anthropomorphic scene.
    """
    seed_values = list(seeds if seeds is not None else DEFAULT_SEEDS)
    scale_values = list(
        perturb_scales if perturb_scales is not None else DEFAULT_PERTURB_SCALES
    )
    records = _collect_records(
        hidden_size=hidden_size,
        steps=steps,
        perturb_step=perturb_step,
        seeds=seed_values,
        perturb_scales=scale_values,
        device=device,
        route_metric_keys=route_metric_keys,
        substrate_kwargs_by_name=substrate_kwargs_by_name or {},
    )
    records.sort(key=_record_sort_key)
    projection_features = _projection_feature_names(route_metric_keys)
    coords, projection = _project_records(records, projection_features)

    points: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {"runs": []}

    for record, coord in zip(records, coords):
        point = {
            "point_id": record["point_id"],
            "x": _clean_float(coord[0]),
            "y": _clean_float(coord[1]),
            "z": _clean_float(coord[2]),
            "step": record["step"],
            "seed": record["seed"],
            "substrate": record["substrate"],
            "run_kind": record["run_kind"],
            "perturb_scale": record["perturb_scale"],
        }
        points.append(point)
        metrics.append({"point_id": record["point_id"], **record["metrics"]})
        if record["is_perturb_step"]:
            events.append(
                {
                    "event_kind": "perturbation",
                    "point_id": record["point_id"],
                    "step": record["step"],
                    "seed": record["seed"],
                    "substrate": record["substrate"],
                    "run_kind": record["run_kind"],
                    "perturb_scale": record["perturb_scale"],
                    "x": point["x"],
                    "y": point["y"],
                    "z": point["z"],
                }
            )
        if record["is_final_step"]:
            events.append(
                {
                    "event_kind": "final_state",
                    "point_id": record["point_id"],
                    "step": record["step"],
                    "seed": record["seed"],
                    "substrate": record["substrate"],
                    "run_kind": record["run_kind"],
                    "perturb_scale": record["perturb_scale"],
                    "x": point["x"],
                    "y": point["y"],
                    "z": point["z"],
                }
            )

    for run in _summarize_runs(records):
        summaries["runs"].append(run)
    summaries["comparison"] = _compact_comparison(
        hidden_size=hidden_size,
        steps=steps,
        perturb_step=perturb_step,
        seeds=seed_values,
        device=device,
        substrate_kwargs_by_name=substrate_kwargs_by_name or {},
    )

    payload = {
        "metadata": {
            "substrates": list(DEFAULT_SUBSTRATES),
            "substrate_labels": {
                CURRENT_TARGET: "active experiment",
                CURRENT_COMPARISON: "comparison",
            },
            "seeds": seed_values,
            "steps": steps,
            "hidden_size": hidden_size,
            "projection_method": "deterministic_pca_on_metric_vectors",
            "projection_features": projection_features,
            "perturb_step": perturb_step,
            "perturb_scales": scale_values,
            "perturb_mode": "noise",
            "schema_version": 1,
            "viewer_note": "fixed-point convergence is shown as geometry, not as failure",
        },
        "projection": projection,
        "points": points,
        "metrics": metrics,
        "events": events,
        "summaries": summaries,
    }

    run_dir = Path(out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "trajectory_3d.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return out_path


def _collect_records(
    hidden_size: int,
    steps: int,
    perturb_step: int,
    seeds: list[int],
    perturb_scales: list[float],
    device: str,
    route_metric_keys: tuple[str, ...],
    substrate_kwargs_by_name: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for substrate in DEFAULT_SUBSTRATES:
        for seed in seeds:
            clean = trajectory_map(
                substrate,
                hidden_size=hidden_size,
                steps=steps,
                seed=seed,
                device=device,
                substrate_kwargs=substrate_kwargs_by_name.get(substrate),
            )
            records.extend(
                _records_from_mapping(
                    substrate=substrate,
                    seed=seed,
                    run_kind="clean",
                    perturb_scale=None,
                    perturb_step=perturb_step,
                    mapping=clean,
                    route_metric_keys=route_metric_keys,
                )
            )
            for scale in perturb_scales:
                perturbed = trajectory_map(
                    substrate,
                    hidden_size=hidden_size,
                    steps=steps,
                    seed=seed,
                    perturb_step=perturb_step,
                    perturb_scale=scale,
                    device=device,
                    substrate_kwargs=substrate_kwargs_by_name.get(substrate),
                )
                records.extend(
                    _records_from_mapping(
                        substrate=substrate,
                        seed=seed,
                        run_kind="perturbed",
                        perturb_scale=scale,
                        perturb_step=perturb_step,
                        mapping=perturbed,
                        route_metric_keys=route_metric_keys,
                    )
                )
    return records


def _records_from_mapping(
    substrate: str,
    seed: int,
    run_kind: str,
    perturb_scale: float | None,
    perturb_step: int,
    mapping: dict[str, Any],
    route_metric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    summary = dict(mapping["summary"])
    rows: list[dict[str, Any]] = []
    trajectory = list(mapping["trajectory"])
    for step in trajectory:
        metrics = _metrics_from_step(step, route_metric_keys)
        point_id = _point_id(
            substrate=substrate,
            seed=seed,
            run_kind=run_kind,
            perturb_scale=perturb_scale,
            step=int(step["step"]),
        )
        rows.append(
            {
                "point_id": point_id,
                "substrate": substrate,
                "seed": seed,
                "run_kind": run_kind,
                "perturb_scale": perturb_scale,
                "step": int(step["step"]),
                "metrics": metrics,
                "summary": summary,
                "is_perturb_step": run_kind == "perturbed"
                and int(step["step"]) == perturb_step,
                "is_final_step": int(step["step"]) == len(trajectory),
            }
        )
    return rows


def _metrics_from_step(
    step: dict[str, Any],
    route_metric_keys: tuple[str, ...],
) -> dict[str, Any]:
    route_metrics = dict(step.get("route_metrics") or {})
    metrics = {key: _clean_float(step.get(key, 0.0)) for key in CORE_METRICS}
    metrics["route_metrics"] = {
        key: _clean_float(route_metrics.get(key, 0.0)) for key in route_metric_keys
    }
    return metrics


def _project_records(
    records: list[dict[str, Any]],
    projection_features: list[str],
) -> tuple[list[tuple[float, float, float]], dict[str, Any]]:
    if not records:
        return [], {"explained_variance_ratio": [0.0, 0.0, 0.0], "axis_loadings": []}

    vectors = np.asarray([_vector_from_metrics(record["metrics"]) for record in records], dtype=float)
    means = vectors.mean(axis=0)
    stds = vectors.std(axis=0)
    safe_stds = np.where(stds < 1e-12, 1.0, stds)
    centered = (vectors - means) / safe_stds

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(3, vt.shape[0])]
    projected = centered @ components.T if components.size else np.zeros((len(records), 0))
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))

    denom = float(np.sum(singular_values ** 2))
    explained = [0.0, 0.0, 0.0]
    if denom > 0.0:
        for idx, value in enumerate((singular_values[:3] ** 2) / denom):
            explained[idx] = _clean_float(value)

    axis_loadings = []
    for axis_idx, component in enumerate(components[:3]):
        ranked = sorted(
            (
                {
                    "feature": projection_features[feature_idx],
                    "weight": _clean_float(weight),
                    "abs_weight": _clean_float(abs(weight)),
                }
                for feature_idx, weight in enumerate(component)
            ),
            key=lambda item: item["abs_weight"],
            reverse=True,
        )
        axis_loadings.append({"axis": axis_idx + 1, "top_features": ranked[:8]})

    return [tuple(float(value) for value in row[:3]) for row in projected], {
        "explained_variance_ratio": explained,
        "axis_loadings": axis_loadings,
    }


def _vector_from_metrics(metrics: dict[str, Any]) -> list[float]:
    vector = [_clean_float(metrics[key]) for key in CORE_METRICS]
    route_metrics = metrics.get("route_metrics") or {}
    vector.extend(_clean_float(route_metrics[key]) for key in sorted(route_metrics))
    return vector


def _projection_feature_names(route_metric_keys: tuple[str, ...]) -> list[str]:
    return list(CORE_METRICS) + [
        f"route_metrics.{key}" for key in sorted(route_metric_keys)
    ]


def _summarize_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for record in records:
        scale_key = "clean" if record["perturb_scale"] is None else str(record["perturb_scale"])
        key = (record["substrate"], record["seed"], record["run_kind"], scale_key)
        if key not in summaries:
            summaries[key] = {
                "substrate": record["substrate"],
                "seed": record["seed"],
                "run_kind": record["run_kind"],
                "perturb_scale": record["perturb_scale"],
                "summary": _compact_summary(record["summary"]),
            }
    return [summaries[key] for key in sorted(summaries)]


def _compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "attractor_type",
        "interior_class",
        "final_norm",
        "mean_norm",
        "mean_delta",
        "mean_coherence",
        "covariance_rank",
        "flow_dimension",
        "compression_ratio",
        "mean_fast_norm",
        "mean_slow_norm",
        "mean_message_norm",
        "max_fast_norm",
        "max_slow_norm",
        "max_message_norm",
    )
    return {key: _json_safe(summary.get(key)) for key in keep if key in summary}


def _compact_comparison(
    hidden_size: int,
    steps: int,
    perturb_step: int,
    seeds: list[int],
    device: str,
    substrate_kwargs_by_name: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if substrate_kwargs_by_name:
        return {
            "aggregate": None,
            "note": (
                "Skipped legacy compare_native_v9_vs_v8 aggregate because this "
                "export uses per-substrate kwargs."
            ),
            "substrate_kwargs_by_name": _json_safe(substrate_kwargs_by_name),
        }
    comparison = compare_native_v9_vs_v8(
        hidden_size=hidden_size,
        steps=steps,
        perturb_step=perturb_step,
        seeds=seeds,
        device=device,
    )
    return {
        "hidden_size": comparison["hidden_size"],
        "steps": comparison["steps"],
        "seeds": comparison["seeds"],
        "aggregate": _json_safe(comparison["aggregate"]),
    }


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    run_order = 0 if record["run_kind"] == "clean" else 1
    scale = -1.0 if record["perturb_scale"] is None else float(record["perturb_scale"])
    return (record["substrate"], record["seed"], run_order, scale, record["step"])


def _point_id(
    substrate: str,
    seed: int,
    run_kind: str,
    perturb_scale: float | None,
    step: int,
) -> str:
    scale = "clean" if perturb_scale is None else str(perturb_scale)
    return f"{substrate}|seed={seed}|{run_kind}|scale={scale}|step={step}"


def _clean_float(value: Any) -> float:
    numeric = float(value)
    if math.isfinite(numeric):
        return numeric
    return 0.0


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, tuple):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _clean_float(value)
    if isinstance(value, float):
        return _clean_float(value)
    return value
