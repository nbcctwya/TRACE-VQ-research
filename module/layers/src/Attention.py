"""Compatibility import; implementation lives in trace_vq.models.components.legacy.attention."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.models.components.legacy.attention")
_sys.modules[__name__] = _implementation
