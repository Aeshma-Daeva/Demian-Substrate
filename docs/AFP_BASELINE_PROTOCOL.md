# Accumulating Fixed-Point Baseline Protocol

Status: experimental protocol. This document defines measurements; it does not
claim that one architecture is superior to another.

## Question

Can an exposed recurrent surface become approximately stationary while the
complete recurrent state continues to evolve, retain history, and influence
what happens under the same future input?

The working name for that regime is **accumulating fixed point (AFP)**.

A strict mathematical fixed point of the complete recurrent state would require

```text
F(z*) = z*
```

AFP deliberately means something weaker and observable:

```text
G(z[t+1]) ~= G(z[t])
while
z[t+1] != z[t]
```

where `z` is the complete recurrent state and `G` is an exposed surface or
readout. Therefore AFP should be described as an **observable/projected
fixed-point regime with continuing latent dynamics**, not as a full-state fixed
point.

## Why two surfaces are measured

The protocol records both:

1. **native surface** — the architecture's own exposed recurrent surface;
2. **common readout** — the same low-dimensional observation procedure applied
   to every architecture.

For Demian v1, the native surface is produced from `fast`, `message`,
`carrier`, and `gate`, while the full latent state contains all six channels:
`fast`, `slow`, `control`, `message`, `carrier`, and `gate`.

For vanilla RNN and GRU, the native hidden state is also their native surface,
so a native-surface AFP is structurally difficult by definition. The common
readout still allows the broader projected-dynamics question to be asked fairly.
LSTM additionally has cell state that is not identical to its exposed hidden
state.

This distinction prevents the baseline comparison from baking the answer into
the definition.

## Matched protocol

The experiment currently compares:

- vanilla tanh RNN;
- GRU;
- LSTM;
- Demian v1.

All architectures receive exactly the same bounded temporal input histories.
The histories have different prefixes and a shared final drive segment, then
all inputs are clamped to zero for a settling window.

The standard recurrent baselines are assigned integer hidden sizes that minimize
the difference between their trainable parameter count and the Demian v1
parameter count. This is approximate matching, not proof of equal effective
capacity.

The common readout is a deterministic orthogonal projection generated from a
fixed seed. No architecture is trained to satisfy the AFP criterion.

## Measurements

For every settling step the probe records RMS-normalized movement of:

- native surface;
- common readout;
- complete latent state.

For the final analysis window it reports:

- mean surface velocity;
- mean readout velocity;
- mean latent velocity;
- relative velocity (movement divided by state scale);
- surface/readout path length;
- latent path length;
- latent-to-surface velocity ratios.

A default AFP candidate requires:

```text
surface_relative_velocity <= surface_rel_epsilon
latent_relative_velocity >= latent_rel_floor
latent_relative_velocity / surface_relative_velocity >= ratio_floor
```

The thresholds are explicit configuration values and must be sensitivity-tested
before a scientific claim is promoted.

## History collision and continuation test

Each architecture is run from the same initial recurrent state across many
different histories. At the end of settling, the probe searches for histories
whose common readouts are close while their complete latent states differ.

Among the closest readout pairs it selects a high latent/readout-gap pair and
continues both states under:

1. identical zero future input;
2. an identical deterministic probe impulse followed by zeros.

The resulting readout gaps are descriptive evidence of path dependence. They
are **not by themselves a perfect causal control**, because the pre-probe
readouts are only near-matched, not mathematically identical.

For Demian, stronger causal controls already exist separately: full-capsule
resume versus surface-only resume and per-channel state surgery.

## Relationship to capsules

Capsulization is not the definition of AFP.

- AFP is a dynamical observation/hypothesis.
- A capsule is an operational state-preservation mechanism.

Capsules matter experimentally because they preserve the complete recurrent
state `z[t]`, allowing two states with similar surfaces but different interiors
to be frozen, resumed, ablated, or compared under the same continuation.

## Run

From the Demian-Substrate repository:

```bash
python -m development.afp_baselines --output-dir outputs/afp_baselines
```

A quick smoke run:

```bash
python -m development.afp_baselines \
  --demian-hidden-size 8 \
  --readout-dim 4 \
  --histories 8 \
  --drive-steps 16 \
  --shared-tail-steps 4 \
  --settle-steps 16 \
  --analysis-window 4 \
  --probe-steps 8 \
  --output-dir /tmp/afp-smoke
```

Outputs:

- `afp_baseline_results.json` — full configuration, per-history metrics,
  parameter matching, pair search, and continuation traces;
- `afp_baseline_summary.csv` — compact architecture-level summary.

## Before promoting a result

A publishable rerun should include at minimum:

- multiple model seeds, not only multiple input histories;
- threshold-sensitivity analysis;
- matched parameter-budget and matched-state-dimension variants;
- confidence intervals across seeds;
- exact preservation of raw JSON outputs;
- a preregistered definition of the AFP criterion;
- comparison against a deliberately constructed full-state fixed-point control;
- Demian capsule/state-surgery controls for causal relevance;
- no claim that a qualitative AFP label proves memory, cognition, awareness, or
  consciousness.

The intended scientific claim is narrower: an observable surface can be nearly
stationary while unobserved recurrent degrees of freedom continue to move, and
those internal differences can be tested for future consequence.
