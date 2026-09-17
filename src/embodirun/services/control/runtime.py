"""Compatibility alias for the canonical control model loop."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.model_loop")
_sys.modules[__name__] = _canonical
