# Glossary

Last updated: 2026-05-12

Purpose: map Demian working vocabulary to standard machine-learning and
dynamical-systems terminology. These terms are operational labels for code,
artifacts, and experiments; they are not claims of biological, cognitive, or
human-like status.

| Demian term | Standard translation | Short definition | Where used |
| --- | --- | --- | --- |
| Substrate | Recurrent dynamical system or architecture family | The stateful system being stepped, perturbed, ablated, and compared. | [README.md](../README.md), [SUBSTRATE_ANATOMY.md](SUBSTRATE_ANATOMY.md) |
| Native substrate | Custom recurrent architecture | A substrate whose state owners and routes are designed directly rather than inherited from a standard cell. | [RESEARCH_LINEAGE.md](RESEARCH_LINEAGE.md), [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md) |
| Channel | Factorized state component | A named component of recurrent state, such as `fast`, `slow`, `control`, `message`, `carrier`, or `gate`. | [SUBSTRATE_ANATOMY.md](SUBSTRATE_ANATOMY.md), [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) |
| Surface | Readout state or observable output vector | The exposed vector used for comparison, visualization, perturbation, or replay. It is not assumed to contain the full computation. | [CAPSULE_CONTINUITY.md](CAPSULE_CONTINUITY.md), [CLAIMS.md](CLAIMS.md#c19-v9-and-v9-five-channel-resume-from-internal-state-capsules-while-surface-only-replay-fails) |
| Capsule continuity | Full internal-state continuation | A pause/resume probe comparing uninterrupted continuation against full internal-state resume and surface-only replay. | [CAPSULE_CONTINUITY.md](CAPSULE_CONTINUITY.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Internal-state capsule | Serialized recurrent state plus model body when applicable | The complete internal state used to test whether a trajectory continues after pause/resume. Current claims are full-state claims, not compression claims. | [CLAIMS.md](CLAIMS.md#c19-v9-and-v9-five-channel-resume-from-internal-state-capsules-while-surface-only-replay-fails) |
| Regime | Attractor or trajectory class | A descriptive class such as fixed point, accumulating fixed point, bounded strange, edge of chaos, or limit cycle. | [CLAIMS.md](CLAIMS.md), [data/INDEX.md](../data/INDEX.md) |
| Fixed-point surface | Stable readout with possible hidden internal dynamics | A condition where exposed output appears fixed while internal channels may keep changing. | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md), [CLAIMS.md](CLAIMS.md#c16-the-v9-five-channel-evolutionary-archive-preserves-multiple-bounded-regimes-including-accumulating-fixed-point-behavior) |
| Bounded strange | Bounded non-periodic trajectory behavior | A qualitative dynamical regime used as an inspection label, not a formal proof of chaos. | [CLAIMS.md](CLAIMS.md#c16-the-v9-five-channel-evolutionary-archive-preserves-multiple-bounded-regimes-including-accumulating-fixed-point-behavior) |
| Release gate | Sparse route-opening control signal | A learned or manual gate that modulates message/carrier coupling in the v9 five-channel scaffold. | [SUBSTRATE_ANATOMY.md](SUBSTRATE_ANATOMY.md), [data/INDEX.md](../data/INDEX.md) |
| Duty cycle | Fraction of steps with gate activity | A summary statistic for how often a release or gate condition is active. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md), [PUBLISHMENT.md](PUBLISHMENT.md) |
| Sparse release | Low-duty route opening | A release-gate phenotype selected in some predecessor runs; current evidence does not show stable cross-seed sparse delayed release. | [CLAIMS.md](CLAIMS.md#c18-the-v100-predecessor-run-selected-against-flood-while-preserving-heritable-sparse-release-for-demian-v1) |
| Routes-disabled | Causal ablation with route effects removed | An intervention that disables release-mediated routes while leaving other model structure available. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md), [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Gain-zero | Causal ablation with release gain set to zero | A diagnostic intervention testing whether divergence persists without explicit additive release gain. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md) |
| Channel-disabled | Per-channel causal ablation | An intervention that zeros a named state channel after each step and feeds the clamped state forward. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md), [data/INDEX.md](../data/INDEX.md) |
| Track A | Engineered-target search track | Search pressure aimed at specified target properties such as delayed sparse release. | [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) |
| Track B | Native-emergence search track | Search pressure that promotes internally useful mechanisms even when they do not satisfy the engineered target. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md) |
| Gate-State Causal Propagation | Replicated internal mechanism | A Track B mechanism where gate/release-state history changes downstream dynamics even under gain-zero diagnostics, with message/carrier necessity and capsule evidence. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md#gate-state-causal-propagation), [CLAIMS.md](CLAIMS.md#c20-gate-state-causal-propagation-replicates-in-track-b-native-emergence-runs) |
| Demian v1 | Next design synthesis | A named prototype direction that makes gate state explicit. It is a design consequence of current evidence, not a completed empirical result. | [NATIVE_MECHANISMS.md](NATIVE_MECHANISMS.md#explicit-gate-state-prototype), [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md#synthesis-demian-v1) |

