"""Minimal substrate compatibility package for Demian v1 runtime."""

from development.substrates.legacy import DemianNativeV9Substrate
from development.substrates.runtime import SelfLoopRunner

__all__ = ["DemianNativeV9Substrate", "SelfLoopRunner"]
