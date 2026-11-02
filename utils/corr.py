"""Compatibility import; implementation lives in trace_vq.evaluation.correlation."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.evaluation.correlation")
_sys.modules[__name__] = _implementation
