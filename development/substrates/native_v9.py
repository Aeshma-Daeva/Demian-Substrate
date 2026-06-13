"""Focused import surface for ``demian_native_v9``.

Implementation still lives in ``development.substrate_lab`` until the legacy
inheritance chain is split. For edits, jump directly to:

    rg -n "class DemianNativeV9Substrate|compare_native_v9_vs_v8" development/substrate_lab.py
"""

from development.substrates.legacy import DemianNativeV9Substrate

SUBSTRATE_NAME = "demian_native_v9"
SOURCE_SYMBOL = "DemianNativeV9Substrate"
COMPARISON_HELPER = "compare_native_v9_vs_v8"

__all__ = ["COMPARISON_HELPER", "DemianNativeV9Substrate", "SOURCE_SYMBOL", "SUBSTRATE_NAME"]
