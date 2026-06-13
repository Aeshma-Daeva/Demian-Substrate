"""Compatibility imports from the legacy substrate lab.

This module is the extraction boundary for active substrate work. It keeps
current code from importing the 10k+ line ``development.substrate_lab`` module
directly while preserving the existing implementations until classes and
helpers can be split safely. The active runner implementation already lives in
``development.substrates.runtime``; this module injects the legacy substrate
registry so summaries keep stable substrate names.
"""

from __future__ import annotations

from development.substrate_lab import (
    DemianNativeV74Substrate,
    DemianNativeV8Substrate,
    DemianNativeV9Substrate,
    SUBSTRATE_REGISTRY,
    _make_substrate,
    compare_native_v9_vs_v8,
    list_substrate_specs,
    trajectory_map,
)
from development.substrates.runtime import SelfLoopRunner as RuntimeSelfLoopRunner


class SelfLoopRunner(RuntimeSelfLoopRunner):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("substrate_registry", SUBSTRATE_REGISTRY)
        super().__init__(*args, **kwargs)

__all__ = [
    "DemianNativeV74Substrate",
    "DemianNativeV8Substrate",
    "DemianNativeV9Substrate",
    "SelfLoopRunner",
    "_make_substrate",
    "compare_native_v9_vs_v8",
    "list_substrate_specs",
    "trajectory_map",
]
