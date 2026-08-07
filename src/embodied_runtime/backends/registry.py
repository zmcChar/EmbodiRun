"""Thread-safe registry for heterogeneous runtime backends."""

from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from embodied_runtime.models.package import ModelPackage

from .device import DeviceInfo
from .interfaces import Backend
from .support import SupportReport


class BackendRegistry:
    """Register and discover model-agnostic hardware backends.

    The distributed layer may eventually advertise the resulting devices over
    the network.  This registry deliberately stays local: it owns backend
    implementations, not node leases, routing, or failover.
    """

    def __init__(self, backends: Iterable[Backend] = ()) -> None:
        self._lock = RLock()
        self._backends: dict[str, Backend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: Backend, *, replace: bool = False) -> None:
        name = backend.name.strip()
        if not name:
            raise ValueError("backend.name must not be empty")
        with self._lock:
            if name in self._backends and not replace:
                raise ValueError(f"backend {name!r} is already registered")
            self._backends[name] = backend

    def unregister(self, name: str) -> Backend:
        with self._lock:
            try:
                return self._backends.pop(name)
            except KeyError as error:
                raise KeyError(f"unknown backend {name!r}") from error

    def get(self, name: str) -> Backend:
        with self._lock:
            try:
                return self._backends[name]
            except KeyError as error:
                available = ", ".join(sorted(self._backends)) or "<none>"
                raise KeyError(
                    f"unknown backend {name!r}; registered backends: {available}"
                ) from error

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._backends))

    def probe(self) -> tuple[DeviceInfo, ...]:
        """Return all locally visible devices in deterministic backend order."""

        with self._lock:
            backends = tuple(self._backends[name] for name in sorted(self._backends))
        devices: list[DeviceInfo] = []
        for backend in backends:
            devices.extend(backend.probe())
        return tuple(devices)

    def support_matrix(
        self,
        package: ModelPackage,
        devices: Iterable[DeviceInfo] | None = None,
    ) -> dict[tuple[str, str], SupportReport]:
        """Evaluate ``package`` against devices without compiling or loading it."""

        candidates = tuple(self.probe() if devices is None else devices)
        reports: dict[tuple[str, str], SupportReport] = {}
        for device in candidates:
            backend = self.get(device.backend)
            reports[(device.backend, device.device_id)] = backend.supports(package, device)
        return reports

    def select(
        self,
        package: ModelPackage,
        devices: Iterable[DeviceInfo] | None = None,
    ) -> tuple[Backend, DeviceInfo, SupportReport]:
        """Return the first supported backend/device pair.

        Routing policy belongs to ``distributed``. This deterministic helper is only a
        local bootstrap mechanism for examples and tests.
        """

        candidates = tuple(self.probe() if devices is None else devices)
        rejected: list[str] = []
        for device in candidates:
            backend = self.get(device.backend)
            report = backend.supports(package, device)
            if report.supported:
                return backend, device, report
            reasons = "; ".join(report.reasons) or "unsupported"
            rejected.append(f"{device.backend}/{device.device_id}: {reasons}")
        detail = " | ".join(rejected) or "no devices were discovered"
        raise LookupError(f"no registered backend supports {package.spec.model_id!r}: {detail}")
