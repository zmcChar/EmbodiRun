"""Structured placeholder for a future Ascend/CANN backend."""

from __future__ import annotations

import importlib.util

from embodied_runtime.models.package import ModelPackage

from ..artifact import ArtifactVariant
from ..compile import CompileOptions
from ..device import DeviceInfo
from ..errors import UnsupportedBackendError
from ..interfaces import BackendSession
from ..support import SupportReport


class AscendBackend:
    """Report SDK and implementation status without importing vendor packages."""

    name = "ascend"
    _sdk_modules = ("torch_npu", "acl")

    def _installed_modules(self) -> tuple[str, ...]:
        return tuple(name for name in self._sdk_modules if importlib.util.find_spec(name))

    def probe(self) -> tuple[DeviceInfo, ...]:
        # Device enumeration is intentionally deferred until a real CANN adapter
        # exists.  Guessing device IDs would make distributed registration unsafe.
        return ()

    def supports(self, package: ModelPackage, device: DeviceInfo) -> SupportReport:
        del package, device
        installed = self._installed_modules()
        if not installed:
            return SupportReport.no("Ascend SDK is unavailable (expected torch_npu or acl)")
        return SupportReport.no(
            "Ascend SDK was detected, but compile/load support is not implemented"
        )

    def compile(
        self,
        package: ModelPackage,
        device: DeviceInfo,
        options: CompileOptions,
    ) -> ArtifactVariant:
        del package, device, options
        raise UnsupportedBackendError(
            "Ascend compile support is a declared extension point, not implemented"
        )

    def load(self, artifact: ArtifactVariant) -> BackendSession:
        del artifact
        raise UnsupportedBackendError(
            "Ascend load support is a declared extension point, not implemented"
        )
