"""Compatibility alias for :mod:`embodirun.application.auth`."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.auth")
_sys.modules[__name__] = _canonical
