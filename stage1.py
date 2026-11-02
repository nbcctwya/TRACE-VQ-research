"""Compatibility CLI; implementation: trace_vq.cli.stage1."""
import importlib as _importlib
import sys as _sys
_implementation = _importlib.import_module("trace_vq.cli.stage1")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
