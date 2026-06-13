"""Focused import surface for ``demian_native_v8``.

Use this as the v9 comparison entry point. The implementation currently lives
in ``development.substrate_lab`` because v8 inherits from older native routes.
"""

from development.substrates.legacy import DemianNativeV8Substrate

SUBSTRATE_NAME = "demian_native_v8"
SOURCE_SYMBOL = "DemianNativeV8Substrate"

__all__ = ["DemianNativeV8Substrate", "SOURCE_SYMBOL", "SUBSTRATE_NAME"]
