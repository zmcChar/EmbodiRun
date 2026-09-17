"""Compatibility facade; canonical implementation lives in ``embodirun.devices.observations.producer``."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.devices.observations.producer")
_sys.modules[__name__] = _canonical
