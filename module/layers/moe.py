"""Compatibility import; implementation lives in trace_vq.models.experts.shared_routed."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.models.experts.shared_routed")
_sys.modules[__name__] = _implementation
