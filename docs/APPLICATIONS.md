# Applications and research consumers

Demian Substrate is the public runtime boundary. The projects below either use
that runtime directly or investigate related recurrent-state mechanisms. Their
results do not become runtime capabilities merely because they share concepts
or channel names.

| Consumer | Input | Question | Publication status |
| --- | --- | --- | --- |
| Acoustic probes in this repository | Synthetic or local PCM-derived, non-semantic features | How does a bounded input perturbation change the recurrent trajectory? | Public source and dated virtual validation |
| EEG observer experiment | EEG-derived coupling sequences | Do input order, checkpoint state, and controlled perturbations alter the observer trajectory? | Aggregate case summary in [Demian Lab](https://github.com/Aeshma-Daeva/Demian-Lab); full adapter and data remain local |
| Wellbore Geo adapter | Depth-ordered well-log measurements | Can sequential measurements produce auxiliary recurrent diagnostics beside conventional baselines? | Case description in [Demian Lab](https://github.com/Aeshma-Daeva/Demian-Lab); no predictive-lift claim |
| Zenith observer work | Typed events from a local Red/Blue simulation | Can recurrent observation be separated from execution authority and referee evidence? | Local research direction; not part of this package or a public Zenith release |

## Shared experimental discipline

Across these cases, a useful result needs more than an interesting trajectory:

1. identify the input transformation;
2. preserve the exact checkpoint boundary;
3. compare full-state continuation with a relevant control;
4. ablate the proposed mechanism;
5. report negative results and environment limitations;
6. keep application claims separate from runtime behavior.

This repository supports the runtime and software controls. Domain validity,
predictive usefulness, and external deployment each require their own evidence.
