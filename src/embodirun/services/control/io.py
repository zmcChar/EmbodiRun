"""Compatibility facade; canonical implementation lives in ``embodirun.devices.execution.io``."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.devices.execution.io")
_sys.modules[__name__] = _canonical
