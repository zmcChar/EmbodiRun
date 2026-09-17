"""Compatibility alias for the canonical application API facade."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.api")
_sys.modules[__name__] = _canonical
