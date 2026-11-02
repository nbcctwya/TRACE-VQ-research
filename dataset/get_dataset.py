"""Compatibility import; implementation lives in trace_vq.data.prepare."""
import importlib as _importlib
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
_implementation = _importlib.import_module("trace_vq.data.prepare")
if __name__ == "__main__":
    _implementation.main()
else:
    _sys.modules[__name__] = _implementation
