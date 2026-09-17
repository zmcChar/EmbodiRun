"""Legacy module alias for the canonical VVLA Wireless backend."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.model_services.backends.vvla.wireless")
_sys.modules[__name__] = _canonical
