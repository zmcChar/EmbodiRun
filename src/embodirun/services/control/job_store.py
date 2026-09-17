"""Compatibility alias for :mod:`embodirun.application.job_store`."""
from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.application.job_store")
_sys.modules[__name__] = _canonical
