"""Compatibility import; implementation lives in trace_vq.utils.wandb."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.utils.wandb")
_sys.modules[__name__] = _implementation
