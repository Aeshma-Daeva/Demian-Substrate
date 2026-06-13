# Experiment Naming

Last updated: 2026-05-11

## Purpose

Keep substrate names, artifact names, and future experiment names separate.
This avoids treating every saved run as a new architecture generation.

## Naming Layers

| Layer | Use | Example |
| --- | --- | --- |
| Substrate scaffold | Code architecture being tested | `v9 five-channel` |
| Canonical baseline | Stable implementation class line | `demian_native_v9` |
| Artifact/run label | Reproducible saved experiment output | `v10.0-frozen-evolution` |
| Custom-substrate program | Next design generation | `Demian v1` / `demian-v1` |

## Current Names

- `demian_native_v9` is the canonical three-channel baseline.
- `v9 five-channel` is the active scaffold with `fast`, `slow`, `control`,
  `message`, and `carrier`.
- `v10.0-frozen-evolution` is a saved predecessor evidence run on the v9
  five-channel scaffold. It is not a `DemianNativeV10Substrate`.
- `Demian v1` is the next named custom-substrate program. It should use the v9
  five-channel and v10.0 evidence to design a cleaner first substrate rather
  than continuing numerical version inflation.

## Next Experiment Names

Use these names for new work unless the experiment is only a rerun of an old
artifact:

| Name | Slug | Meaning |
| --- | --- | --- |
| Demian v1 | `demian-v1` | First custom-substrate program distilled from v9 five-channel evidence. |
| Demian v1 cross-eval | `demian-v1-cross-eval` | Held-out seed evaluation of the v10.0 predecessor winner and near-winners. |
| Demian v1 release-causality | `demian-v1-release-causality` | Tests whether sparse release has local geometric consequence, not just low duty. |
| Demian v1 anatomy-freeze | `demian-v1-anatomy-freeze` | Freezes channel roles, metrics, and invalid-regime gates before new evolution. |

## Rules

- Do not create `demian_native_v10` unless there is a new implementation class.
- Do not call a scoring rerun a new substrate version.
- Preserve old artifact paths; add aliases or docs instead of renaming raw data.
- Use `demian-v*` names for custom-substrate programs and `v9-5ch-*` names for
  scaffold-specific probes.
- Promote a new `Demian vN` only when anatomy, code surface, artifacts, and
  falsifiers are all explicit.
