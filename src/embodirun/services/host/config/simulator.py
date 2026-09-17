"""Compatibility alias for :mod:`embodirun.deployment.config.simulator`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.deployment.config.simulator")
_sys.modules[__name__] = _canonical
