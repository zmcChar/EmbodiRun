"""Compatibility alias for :mod:`embodirun.application.direct_execution`."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.direct_execution")
_sys.modules[__name__] = _canonical
