"""Compatibility alias for canonical simulation runtime."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.simulation.runtime")
_sys.modules[__name__] = _canonical
