"""Small workbench for the active v9/v8/v7.4 substrate line.

This module is the intended restart surface for current experiments. It keeps
the operational API compact while preserving compatibility with the larger
``development.substrate_lab`` module.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from development.evolution.artifacts import (
    load_summary,
    v10_summary_digest,
    validate_v10_summary,
)
from development.evolution.current import (
    CURRENT_EVIDENCE_ID,
    CURRENT_EXPERIMENT_ID,
    CURRENT_EXPERIMENT_SUMMARY_PATH,
    current_experiment_metadata,
)
from development.substrates.legacy import (
    DemianNativeV74Substrate,
    DemianNativeV8Substrate,
    DemianNativeV9Substrate,
    _make_substrate,
    compare_native_v9_vs_v8,
)
from development.substrates.trajectory_export import export_current_trajectory_3d

CURRENT_TARGET = "demian_native_v9"
CURRENT_COMPARISON = "demian_native_v8"
CURRENT_BASELINE = "demian_native_v7.4"
CURRENT_SUBSTRATES = (CURRENT_TARGET, CURRENT_COMPARISON, CURRENT_BASELINE)

DEFAULT_V9_V8_RESULT_DIR = Path("data/substrate_lab/v9_v8_compare_20260508")
DEFAULT_V10_SUMMARY_PATH = CURRENT_EXPERIMENT_SUMMARY_PATH
DEFAULT_MANIFEST_PATH = Path("data/MANIFEST.json")

CURRENT_DOCS = (
    Path("docs/WORKING_STATE.md"),
    Path("docs/SUBSTRATE_ANATOMY.md"),
    Path("docs/LABBOOK.md"),
    Path("docs/CLAIMS.md"),
    Path("data/INDEX.md"),
    Path("data/MANIFEST.json"),
)


@dataclass(frozen=True)
class WorkbenchEntry:
    """One active substrate role exposed by the compact workbench."""

    name: str
    role: str
    class_name: str
    source_symbol: str
    implementation_path: str
    notes: str


CURRENT_CLASSES = {
    CURRENT_TARGET: DemianNativeV9Substrate,
    CURRENT_COMPARISON: DemianNativeV8Substrate,
    CURRENT_BASELINE: DemianNativeV74Substrate,
}

CURRENT_WORKBENCH = (
    WorkbenchEntry(
        name=CURRENT_TARGET,
        role="active canonical baseline",
        class_name="DemianNativeV9Substrate",
        source_symbol="DemianNativeV9Substrate",
        implementation_path="development/substrate_lab.py",
        notes="Minimal three-channel v9 baseline: fast, slow, control.",
    ),
    WorkbenchEntry(
        name=CURRENT_COMPARISON,
        role="immediate comparison",
        class_name="DemianNativeV8Substrate",
        source_symbol="DemianNativeV8Substrate",
        implementation_path="development/substrate_lab.py",
        notes="Seven-channel comparison scaffold for the v9 direction.",
    ),
    WorkbenchEntry(
        name=CURRENT_BASELINE,
        role="promoted historical baseline",
        class_name="DemianNativeV74Substrate",
        source_symbol="DemianNativeV74Substrate",
        implementation_path="development/substrate_lab.py",
        notes="Organ-heavy ownership/viability historical baseline.",
    ),
)

CURRENT_ARTIFACTS = {
    "demian_v1_predecessor": DEFAULT_V10_SUMMARY_PATH,
    "v10_frozen_evolution": DEFAULT_V10_SUMMARY_PATH,
    "v9_v8_comparison": DEFAULT_V9_V8_RESULT_DIR / "summary.json",
    "manifest": DEFAULT_MANIFEST_PATH,
}


def current_substrate_specs() -> tuple[dict[str, str], ...]:
    """Return active substrate roles without importing the legacy lab again."""
    return tuple(asdict(entry) for entry in CURRENT_WORKBENCH)


def current_entrypoints() -> dict[str, Any]:
    """Return the compact restart surface for active substrate work."""
    return {
        "docs": [path.as_posix() for path in CURRENT_DOCS],
        "legacy_lab": "development/substrate_lab.py",
        "runtime": "development/substrates/runtime.py",
        "legacy_boundary": "development/substrates/legacy.py",
        "current_experiment": current_experiment_metadata(),
        "substrates": current_substrate_specs(),
        "artifacts": {name: path.as_posix() for name, path in CURRENT_ARTIFACTS.items()},
        "tests": [
            "tests/test_current_substrates.py",
            "tests/test_v9_5ch_evolution.py",
            "tests/test_substrate_lab.py",
        ],
    }


def make_current_substrate(
    substrate_name: str,
    hidden_size: int = 32,
    **substrate_kwargs: float,
):
    """Instantiate one active-line substrate by canonical name."""
    if substrate_name not in CURRENT_SUBSTRATES:
        raise KeyError(
            f"{substrate_name!r} is outside the current workbench: "
            f"{', '.join(CURRENT_SUBSTRATES)}"
        )
    return _make_substrate(substrate_name, hidden_size, substrate_kwargs)


def compare_current_target(
    hidden_size: int = 32,
    steps: int = 128,
    perturb_step: int = 64,
    seeds: list[int] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run the current v9-v8 comparison with explicit defaults."""
    return compare_native_v9_vs_v8(
        hidden_size=hidden_size,
        steps=steps,
        perturb_step=perturb_step,
        seeds=seeds,
        device=device,
    )


def write_v9_v8_comparison(
    out_dir: Path | str = DEFAULT_V9_V8_RESULT_DIR,
    hidden_size: int = 32,
    steps: int = 128,
    perturb_step: int = 64,
    seeds: list[int] | None = None,
    device: str = "cpu",
) -> Path:
    """Run and save the compact v9-v8 comparison artifact."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result = compare_current_target(
        hidden_size=hidden_size,
        steps=steps,
        perturb_step=perturb_step,
        seeds=seeds,
        device=device,
    )
    summary_path = out_path / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    return summary_path


def load_current_v9_v8_summary(
    summary_path: Path | str = DEFAULT_V9_V8_RESULT_DIR / "summary.json",
) -> dict[str, Any]:
    """Load the saved v9-v8 summary without inspecting raw data directories."""
    return json.loads(Path(summary_path).read_text())


def load_latest_evolution_summary(
    summary_path: Path | str = DEFAULT_V10_SUMMARY_PATH,
) -> dict[str, Any]:
    """Load the v10 predecessor summary used as Demian v1 evidence."""
    summary = load_summary(summary_path)
    errors = validate_v10_summary(summary)
    if errors:
        raise ValueError(f"invalid {CURRENT_EVIDENCE_ID} summary: {'; '.join(errors)}")
    return summary


def load_current_experiment_digest(
    summary_path: Path | str = DEFAULT_V10_SUMMARY_PATH,
) -> dict[str, Any]:
    """Load the compact predecessor digest for the current named experiment."""
    digest = v10_summary_digest(load_latest_evolution_summary(summary_path))
    digest["current_experiment"] = CURRENT_EXPERIMENT_ID
    digest["predecessor_evidence"] = CURRENT_EVIDENCE_ID
    return digest


def load_manifest_run(
    run_id: str,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load one run entry from the artifact manifest by id."""
    payload = json.loads(Path(manifest_path).read_text())
    for run in payload.get("runs", []):
        if run.get("id") == run_id:
            return run
    raise KeyError(f"run id not found in manifest: {run_id}")
