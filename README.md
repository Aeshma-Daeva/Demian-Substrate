# Demian Substrate

Demian Substrate is the runtime package for the current Demian v1 recurrent
substrate. It is the part that should be small, importable, testable, and clear:
the code that owns the state, advances it one step, saves it, restores it, and
reports what changed inside.

This is not the lab notebook. The big experiment logs, figures, sweeps, and
failed branches belong in Demian Lab or Demian Archive. This repo keeps the
machine that those experiments pushed toward.

## Plain-English Overview

A normal recurrent model has hidden state. Demian v1 makes that hidden state
less anonymous. Instead of one hidden blob, it keeps named internal channels:

- `fast`: the visible surface state.
- `slow`: longer memory and basin continuity.
- `control`: steering pressure.
- `message`: short-lived perturbation trace.
- `carrier`: slower message accumulation.
- `gate`: a learned internal pressure signal that changes route strength.

The important idea is simple: the visible surface is not the whole system. If
you only save what the surface looks like, you lose the internal context that
decides where the next steps go. A real capsule must save every channel.

## Outside-Lab Observer Test

Demian v1 was also attached to Abraxas, a separate combat-swarm simulation, as
a zero-authority recurrent observer. This was not a "Demian controls the
system" test. It was the opposite: Demian was purposely blocked from choosing
actions so the run could ask a cleaner question.

Can the substrate attach to an outside event stream, keep temporal continuity,
and expose useful recurrent diagnostics without being allowed to cause the
behavior it is measuring?

The Abraxas bridge fed completed combat transitions into Demian after each
combat cycle:

```text
Abraxas transition log
  -> fixed combat feature vector + missing-value mask
  -> folded coupling vector
  -> DemianV1Runtime.step(coupling, strength=0.25)
  -> surface, metrics, and recurrent capsule
  -> Abraxas checkpoint/quorum evidence packet
```

Abraxas did not write directly into Demian's six channels. It supplied one
structured coupling vector. Demian then routed that input through its own
`fast`, `slow`, `control`, `message`, `carrier`, and `gate` state.

Application-level readings were:

| Channel | Abraxas reading |
| --- | --- |
| `fast` | Current tactical surface: latest resources, selected action, execution result, reward. |
| `slow` | Mission continuity: objective progress, active objectives, durable memory priors, capsule continuity. |
| `control` | Intervention pressure: Meta-Red confidence/noise, risk budget, Blue pressure, dispatch validity. |
| `message` | Recent perturbation: action identity, macro identity, casualties, Blue response, zone pressure changes. |
| `carrier` | Accumulated opponent/terrain pattern: threats, monitoring, decoys, blocked routes, spawn pressure. |
| `gate` | Demian-internal relevance pressure and route modulation. |

The result was substantial but bounded:

| Campaign | Result |
| --- | ---: |
| 500-run validation | Demian available in 500/500 runs. |
| 500-run validation | Full-state resume beat surface-only resume in 500/500 runs. |
| 500-run validation | Ordered input differed from shuffled input in 500/500 runs. |
| 500-run validation | Live recurrent state differed from frozen state in 500/500 runs. |
| 1,000-run max calibration | Demian available in 1,000/1,000 runs. |
| 1,000-run max calibration | 12 recurrent steps per run, all classified as `tight_cycle`. |

The useful reading is not "Demian won." Demian did not make tactical choices.
It observed a dynamic external system and preserved enough recurrent structure
for Abraxas to include it in checkpoint evidence about convergence, repetition,
spawn saturation, duplicate transfer keys, and curriculum pressure.

What this proves:

- Demian can run inside a non-lab application loop as an observe-only recurrent
  diagnostic substrate.
- Ordered temporal input is not equivalent to shuffled input.
- Live recurrent state is not equivalent to frozen state.
- Full recurrent capsules carry continuity that the visible surface alone does
  not preserve.

What this does not prove:

- Demian should choose combat actions.
- Demian improves tactical performance.
- Demian has general decision authority.
- A `tight_cycle` result is automatically good; in Abraxas it was useful
  because it lined up with repeated spawn-heavy basin behavior.

## Why This Shape Exists

Demian did not start with this six-channel runtime. The current shape is the
compressed result of earlier recurrent tests:

| Line | What it taught |
| --- | --- |
| Vanilla RNN / GRU / LSTM | Matched baselines often settle into fixed-point behavior. Their surface can look stable while telling us little about richer internal routing. |
| Dual-GRU and early native runs | Fixed-point surfaces can hide useful internal structure, so surface labels alone are weak evidence. |
| `demian_native_v8` / `demian_native_v9` | Fewer named channels made the system easier to inspect and compare. |
| v9 five-channel work | `message` and `carrier` became the main places to test whether perturbation history was being stored and routed. |
| Track B gate-state work | The useful question shifted from "did release open?" to "did internal gate pressure change the future route?" |
| Demian v1 | The gate became a first-class channel instead of a side effect hidden inside release metrics. |

The 2026-05-16 Track B replication manifest selected 27 candidates from
generations 10 through 19. The selected rows had high internal richness
(`0.985` mean) and channel separation (`0.673` mean), but the saved archive rows
could not prove channel dominance by themselves. The locked truth campaign then
made the boundary stricter: Track B did not pass as a strong mechanism claim
(`0/29` strict-profile held-out passes), while state surgery still showed that
swapping `message`/`carrier` often changes continuation (`206` message/carrier
dominant windows versus `118` slow/control dominant windows).

That is why this repo does not claim "solved architecture." It keeps the
runtime anatomy that made the next testable version possible.

## Current Runtime Anatomy

```text
coupling input -> fast

fast -> message
message -> carrier
carrier -> slow

fast + slow + control + message + carrier -> gate
gate -> route modulation

fast + message + carrier + gate -> exposed surface
```

The gate is continuous. It does not inject a separate release vector. It changes
how strongly existing routes act:

- `message -> carrier`
- `carrier -> slow`
- `message/carrier -> fast surface`

So the current question is not "did a rare release event fire?" The current
question is "did gate pressure change the route enough to matter?"

## What The Metrics Mean

The runtime returns a small metric dictionary on every step. Read it like a
dashboard for internal movement:

| Metric | Plain reading |
| --- | --- |
| `message_write_mean` | How much the message channel updated this step. |
| `carrier_write_mean` | How much the carrier channel updated this step. |
| `gate_write_mean` | How much the gate channel wanted to rewrite itself. |
| `message_to_carrier_norm` | Strength of the message-to-carrier route after gate modulation. |
| `carrier_to_slow_norm` | Strength of the carrier-to-slow route after gate modulation. |
| `message_to_fast_norm` / `carrier_to_fast_norm` | How much internal message/carrier state reaches the visible surface. |
| `gate_state_norm` | Size of the gate state. |
| `gate_state_delta` | How much the gate changed this step. |
| `gate_change_duty` | Fraction of gate units that moved past the change threshold. |
| `gate_pressure_mean` | Average gate pressure after sigmoid. |
| `gate_modulation_mean` | Average route-strength multiplier produced by the gate. |

The most important control is resume quality:

- full capsule restore saves all six channels and should continue exactly;
- surface-only restore keeps the visible surface and zeros the internals, so it
  should diverge when the hidden channels matter.

## Ablations

Ablations are the "turn one thing off and see if the future changes" tests.
They keep the story honest.

| Ablation | What it asks |
| --- | --- |
| `gate_disabled` | What happens if the gate is forced to zero? |
| `gate_frozen` | What happens if the gate cannot keep changing? |
| `message_disabled` | Is the short perturbation trace actually needed? |
| `carrier_disabled` | Is accumulated message state actually needed? |
| `slow_disabled` | Is longer continuity doing real work? |
| `control_disabled` | Is steering pressure doing real work? |
| surface-only restore | Can the surface alone reproduce the future, or does the capsule matter? |

## Public API

```python
import torch

from demian_v1 import DemianV1Config, DemianV1Runtime

runtime = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=101))

first = runtime.step(torch.ones(8) * 0.05)
snapshot = runtime.snapshot()

clone = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=101))
clone.restore(snapshot)

surface_control = DemianV1Runtime(DemianV1Config(hidden_size=8, seed=101))
surface_control.restore(snapshot, surface_only=True)
```

`snapshot()` is the capsule boundary. It stores:

- `runtime_id`: stable identity, currently `demian-v1`.
- `config`: construction settings.
- `step_index`: current recurrent step.
- `channels`: `fast`, `slow`, `control`, `message`, `carrier`, `gate`.
- `model_state`: flattened model parameters for reproducible handoff.

## Repository Contents

- `demian_v1/`: stable public runtime and capsule API.
- `development/demian_v1_gate_state.py`: current six-channel gate-state
  prototype.
- `development/substrates/`: current substrate runtime and compatibility
  boundary.
- `legacy/demian_runtime/machine_observables.py`: minimal historical observable
  helpers still used by the current substrate code.
- `tests/`: substrate package tests.
- `docs/`: terminology and substrate anatomy references.

## Verify

Run the full substrate check:

```bash
python -m pytest
```

Run only the public runtime and gate-state checks:

```bash
python -m pytest tests/test_demian_v1_public_api.py tests/test_demian_v1_gate_state.py
```

## Boundary

New package and runtime work belongs here. Broad experiments, sweeps, paper
drafts, generated diagnostic figures, and raw campaign artifacts belong in
Demian Lab. Historical mechanisms and superseded notes belong in Demian
Archive.
