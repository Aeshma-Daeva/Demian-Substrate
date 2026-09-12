# Virtual audio validation freeze — 2026-09-12

This checkpoint freezes the non-physical validation completed before live
microphone acceptance. The repository supports deterministic offline audio,
incremental framing, bounded live-session handoff, optional `sounddevice`
capture, virtual WAV capture, segmented pause/resume, and privacy-preserving
closed-boundary checkpoints.

## Verified evidence

- The complete repository suite passed: 66 tests.
- Focused lint, compilation, and diff checks passed.
- Seven deterministic two-second, 48 kHz signals traversed the virtual capture
  callback, live-session, feature, coupling, recurrent-runtime, and JSONL path:
  silence, 120 Hz tone, 300 Hz tone, quiet-to-loud, sweep, noise burst, and a
  synthetic rhythmic mixture.
- Every session stopped valid with 96,000 accepted samples, 198 complete frames,
  and one explicitly padded tail frame.
- Same-environment reruns reproduced frames and events byte-for-byte. Alternate
  callback partitions reproduced every frame byte-for-byte.
- The virtual WAV backend reads PCM incrementally. Independent 1-second and
  100-second probes kept reads bounded to the configured 509-frame callback;
  traced peak allocations remained approximately 37 KB and 33 KB respectively.
- Pause/resume tests preserve recurrent state and projection identity while
  starting a new segment with fresh framing and acoustic history.
- Closed-boundary checkpoints exclude pending PCM and previous spectral history,
  validate before mutation, and restore into a new segment.

## Scientific boundary

The controlled signals produce distinct acoustic features, coupling vectors,
and recurrent trajectories. In the noise-burst comparison, coupling returned
to the matched silence value while a surface difference remained through the
end of the two-second trace. Silence itself also produces substantial recurrent
evolution, so acoustic effects must be assessed against a matched silence run.

These results demonstrate deterministic signal sensitivity, inspectable
perturbation, and state persistence in this configuration. They do not establish
speech or music understanding, emotion, identity, agency, consciousness, or an
isolated causal role for any named internal channel.

## Reproduce

From the repository root, using a fresh output directory:

```bash
python -m development.virtual_audio_experiment \
  --signal sweep \
  --duration 2 \
  --output-dir /tmp/demian-sweep-reproduction
```

The virtual fixtures are generated deterministically; bulky JSONL traces are
not committed because they are reproducible from the code and tests.

## Deferred acceptance

All enumerated Linux input routes completed without overflow or device failure,
but delivered digital silence. Physical microphone routing, nonzero real input,
real-time scheduling under actual use, unplug behavior, and listening-level
acceptance remain deferred until the microphone is available.
