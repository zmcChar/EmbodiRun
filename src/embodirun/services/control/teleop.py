"""Compatibility facade; canonical implementation lives in ``embodirun.devices.execution.teleop``."""

from importlib import import_module as _import_module
import sys as _sys

_canonical = _import_module("embodirun.devices.execution.teleop")
_sys.modules[__name__] = _canonical


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI test
    raise SystemExit(_canonical.main())
