# Applications and research consumers

Demian Substrate is the public runtime boundary. The projects below either use
that runtime directly or investigate related recurrent-state mechanisms. Their
results do not become runtime capabilities merely because they share concepts
or channel names.

| Consumer | Input | Question | Publication status |
| --- | --- | --- | --- |
| Acoustic probes in this repository | Synthetic or local PCM-derived, non-semantic features | How does a bounded input perturbation change the recurrent trajectory? | Public source and dated virtual validation |
| [Demian EEG](https://github.com/Aeshma-Daeva/Demian-EEG) | Synthetic EEG-like coupling sequences | Do input order and checkpoint completeness alter the observer trajectory? | Public minimal adapter and tests; raw EEG remains excluded |
| [Demian Geo](https://github.com/Aeshma-Daeva/Demian-Geo) | Synthetic depth-ordered well-log measurements | Can sequential measurements produce auxiliary recurrent diagnostics beside conventional baselines? | Public minimal adapter and tests; no predictive-lift claim |
| [Zenith Epistemic Runtime](https://github.com/Aeshma-Daeva/Zenith-Epistemic-Runtime) | Evidence, validity, contradiction, and refresh events | Can belief standing be separated from action authority? | Public typed state machine; not a complete agent or truth-discovery system |

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
