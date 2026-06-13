# Demian Substrate

Clean package boundary for the Demian v1 recurrent substrate runtime.

This repository contains the public `demian_v1` runtime API plus the current
gate-state prototype code it depends on. The `development/` and minimal
`legacy/` files are transitional compatibility surfaces retained so the
substrate tests run from this standalone repo.

## Contents

- `demian_v1/`: stable public runtime and capsule API.
- `development/demian_v1_gate_state.py`: Demian v1 gate-state prototype.
- `development/substrates/`: current substrate runtime and compatibility
  boundary.
- `legacy/demian_runtime/machine_observables.py`: minimal historical observable
  helpers still used by the current substrate code.
- `tests/`: substrate package tests.
- `docs/`: terminology and substrate anatomy references.

## Verify

```bash
python -m pytest
```

## Boundary

New package/runtime work belongs here. Broad experiments, sweeps, paper drafts,
and generated diagnostic figures belong in Demian Lab. Historical mechanisms
and superseded notes belong in Demian Archive.
