"""Command-line entry point for the deterministic offline voice probe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from demian_v1.audio_probe import run_audio_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="uncompressed integer PCM WAV")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--runtime-seed", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--strength", type=float, default=0.15)
    parser.add_argument("--frame-ms", type=float, default=25.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_audio_probe(
        args.input,
        args.output,
        hidden_size=args.hidden_size,
        runtime_seed=args.runtime_seed,
        projection_seed=args.projection_seed,
        strength=args.strength,
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
