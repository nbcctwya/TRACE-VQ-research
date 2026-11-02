"""Compatibility import; implementation lives in trace_vq.training.prediction."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.training.prediction")
_sys.modules[__name__] = _implementation
