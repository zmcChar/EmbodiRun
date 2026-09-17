"""Compatibility alias for :mod:`embodirun.deployment.executor.command`."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.deployment.executor.command")
_sys.modules[__name__] = _canonical
