"""Compatibility alias for :mod:`embodirun.deployment.state`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.deployment.state")
_sys.modules[__name__] = _canonical
