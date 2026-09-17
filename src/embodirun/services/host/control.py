"""Compatibility alias for :mod:`embodirun.deployment.control`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.deployment.control")
_sys.modules[__name__] = _canonical
