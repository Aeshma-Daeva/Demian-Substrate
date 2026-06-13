# Substrate Anatomy

Last updated: 2026-05-11

## Purpose

This is the stable anatomy reference for the active Demian substrate line. It
explains what the channels do, how they route into each other, and how to read
the current v9 five-channel and Demian v1 evidence without opening the full
historical lab first.

Use this with:

- [docs/LABBOOK.md](/home/xenith/demian/docs/LABBOOK.md)
- [docs/CLAIMS.md](/home/xenith/demian/docs/CLAIMS.md)
- [docs/EXPERIMENT_NAMING.md](/home/xenith/demian/docs/EXPERIMENT_NAMING.md)
- [data/INDEX.md](/home/xenith/demian/data/INDEX.md)
- [data/MANIFEST.json](/home/xenith/demian/data/MANIFEST.json)

## Current Channel Roles

The active experimental line uses five channels:

| Channel | Role | Interpretation boundary |
| --- | --- | --- |
| `fast` | Exposed surface state | What the substrate presents as the current trajectory surface. |
| `slow` | Deep continuity state | Longer-horizon basin memory and accumulated route deformation. |
| `control` | Pressure/steering state | Internal pressure that biases gates and release decisions. |
| `message` | Shorter-timescale perturbation trace | Carries structured perturbation information before release. |
| `carrier` | Longer-timescale perturbation trace | Stores slower accumulated message shape and release pressure. |

These are not human-facing roles. They are machine-observable state owners.
Interpret them by routing, timescale, gate behavior, and trajectory effects.

## Canonical v9

Canonical `demian_native_v9` is the minimal three-channel baseline:

```text
fast -> slow
fast + slow -> control
control -> fast
```

Current implementation source:

- [development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py)
- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
- [development/substrates/native_v9.py](/home/xenith/demian/development/substrates/native_v9.py)

Canonical v9 is a diagnostic baseline, not the full active experiment. It
tests whether strict three-channel coupling can produce useful attractor
structure without the larger v8/v7.4 route set.

## Demian v1 On The v9 Scaffold

The next named experiment program is `Demian v1` (`demian-v1`). It runs on the
v9 five-channel scaffold, which extends the v9 direction with explicit
`message` and `carrier` accumulation plus sparse release. The existing
`v10.0-frozen-evolution` artifact is predecessor evidence for this program, not
a separate `DemianNativeV10Substrate` class:

```text
fast/control pressure -> message
message -> carrier
carrier/control pressure -> release gate
release -> fast/slow path geometry
```

Important implementation surface:

- [development/evolution/](/home/xenith/demian/development/evolution)
- [development/probe_v9_message_carrier_strange.py](/home/xenith/demian/development/probe_v9_message_carrier_strange.py)
- [development/evolve_v9_5ch_release.py](/home/xenith/demian/development/evolve_v9_5ch_release.py)
- [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
- [tests/test_v9_5ch_evolution.py](/home/xenith/demian/tests/test_v9_5ch_evolution.py)

The release path should be rare and consequential. Always-on release behaves
like another residual path and is not the target. The current selection target
is instrumental sparsity: release opens rarely enough to avoid flood behavior
and strongly enough to alter bounded path geometry.

## Demian v1 Explicit Gate-State Prototype

The first v1 synthesis prototype makes gate-state propagation an explicit
sixth channel instead of treating it as an emergent side effect of release
state. Its active state owners are:

```text
fast, slow, control, message, carrier, gate
```

The prototype route reading is:

```text
fast -> message -> carrier -> slow
control/message/carrier/slow -> gate
gate -> graded route modulation
```

The gate does not inject a release vector into the state. It modulates existing
message-carrier, carrier-slow, and message/carrier-surface routes. Sparsity is
therefore read as sparse gate-state change or pressure variation, not rare
release-open events.

Implementation surface:

- [development/demian_v1_gate_state.py](/home/xenith/demian/development/demian_v1_gate_state.py)
- [tests/test_demian_v1_gate_state.py](/home/xenith/demian/tests/test_demian_v1_gate_state.py)

## Replicated Gate-State Propagation

Track B native-emergence replications established a separate mechanism from
sparse route-specific release. In `Gate-State Causal Propagation`, original
trajectories diverge from gain-zero and route-disabled variants even when
gain-zero diagnostics confirm zero release strength and zero release-route
output.

The replicated channel pattern is:

```text
message + carrier state -> gate-state propagation
slow state -> supporting continuation memory
surface-only resume -> insufficient continuation
full internal-state resume -> exact continuation
```

Interpretation boundary:

- This is a native internal-state mechanism, not proof that sparse delayed
  release has been solved.
- `message` and `carrier` are the repeated necessary channels in the Track B
  replications.
- `slow` carries supporting continuation state, but the replicated core is the
  message/carrier propagation path.
- Metrics should keep route-specific causal release separate from gain-zero
  gate-state propagation.

## Metric Reading

| Metric | What it reads | Use |
| --- | --- | --- |
| `release_duty_cycle` | Fraction of release-open steps | Detects flood versus sparse release. |
| `release_geometric_event` | Release-local trajectory change | Measures whether release creates geometric consequence. |
| `phase_transition_score` | Phase/event transition structure | Measures whether release aligns with regime change or path reorganization. |
| `internal_richness` | Differentiated internal activity | Checks whether surface-simple behavior hides active internals. |
| `channel_separation` | Channel differentiation | Guards against all channels echoing the same state. |
| `geometric_coherence` | Bounded structured path behavior | Separates coherent bounded motion from incoherent expansion. |
| `mathematical_curiosity` | Structural trajectory interest | Auxiliary shape/richness signal, not a human preference score. |
| `regime_bonus` | Attractor/regime class score | Rewards bounded nontrivial regimes without treating fixed points as failure. |

`boundedness` is currently tracked as a metric but removed from rank scoring
because it was constant in the predecessor v10.0 run. If a future
bound-collapse detector is added,
it should gate scoring or flag invalid candidates rather than add a constant
rank contribution.

## Regime Reading

Current regime labels are behavior classes, not value judgments:

| Regime | Reading |
| --- | --- |
| `surface_fixed_accumulating` | Surface appears fixed while internal channels accumulate structure. |
| `bounded_strange` | Bounded non-periodic trajectory structure. |
| `edge_of_chaos` | Near-boundary behavior requiring matched rerun support before promotion. |
| `limit_cycle_*` | Periodic structure; useful when it is stable and bounded. |
| `unbounded_expanding` | Usually an invalid or low-trust regime unless specifically under study. |
| `collapsed_incoherent` | Low-trust collapse signal. |

Do not dismiss fixed-point behavior without checking internal richness,
message/carrier traces, and release-local geometry.

## Native Lineage Deltas

Use this as ancestry orientation, not as a replacement for artifacts:

| Line | Main delta |
| --- | --- |
| `demian_native_v0` | First native route scaffold with explicit fast/slow/carrier/packet/control routes. |
| `demian_native_v1-v2` | Strengthened slow recruitment and carrier persistence. |
| `demian_native_v3` | Added endogenous tightness control around consolidation versus looseness. |
| `demian_native_v5.x` | Added route-local plasticity, delayed credit, phase-aware anti-locking. |
| `demian_native_v6` | Added endogenous observer/value/actuator control and multi-horizon critic signals. |
| `demian_native_v7.x` | Added ancestry, active/projection/boundary organs, trajectory memory, and metabolic constraints. |
| `demian_native_v7.4` | Added ownership/viability pressure, self-policy gates, quarantine, and topology shadow. |
| `demian_native_v8` | Seven-channel genotype scaffold and immediate comparison line for v9. |
| `demian_native_v9` | Three-channel collapse into `fast`, `slow`, and `control`. |
| v9 five-channel / Demian v1 | Experimental message/carrier/release scaffold on top of the v9 direction. |

The historical implementation still lives primarily in
[development/substrate_lab.py](/home/xenith/demian/development/substrate_lab.py).
Use focused workbench surfaces first and jump into the legacy file only for
targeted symbol ranges.

## Lightweight Workbench

Use [development/substrates/current.py](/home/xenith/demian/development/substrates/current.py)
as the compact code entry point. It exposes:

```text
current_substrate_specs()
current_entrypoints()
make_current_substrate(...)
compare_current_target(...)
load_latest_evolution_summary(...)
load_manifest_run(...)
```

The workbench is intentionally small. It makes active work discoverable without
forcing a broad read of the 10k+ line historical implementation.

[development/substrates/legacy.py](/home/xenith/demian/development/substrates/legacy.py)
is the narrow compatibility boundary for legacy symbols still needed by active
work. New active imports should route through that boundary rather than directly
through `development.substrate_lab`.

[development/substrates/runtime.py](/home/xenith/demian/development/substrates/runtime.py)
owns the active runner and generic run metrics. This is the first extracted code
slice from the historical lab and should be the default location for future
runner-level changes.
