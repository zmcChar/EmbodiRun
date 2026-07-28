"""Structured placeholder for a future Horizon BPU backend."""

from __future__ import annotations

import importlib.util

from embodied_runtime.contracts import (
    ArtifactVariant,
    BackendSession,
    CompileOptions,
    DeviceInfo,
    ModelPackage,
    SupportReport,
    UnsupportedBackendError,
)


class HorizonBackend:
    """Report SDK and implementation status without importing vendor packages."""

    name = "horizon"
    _sdk_modules = ("hbdk4", "hobot_dnn")

    def _installed_modules(self) -> tuple[str, ...]:
        return tuple(name for name in self._sdk_modules if importlib.util.find_spec(name))

    def probe(self) -> tuple[DeviceInfo, ...]:
        return ()

    def supports(self, package: ModelPackage, device: DeviceInfo) -> SupportReport:
        del package, device
        installed = self._installed_modules()
        if not installed:
            return SupportReport.no("Horizon SDK is unavailable (expected hbdk4 or hobot_dnn)")
        return SupportReport.no(
            "Horizon SDK was detected, but compile/load support is not implemented"
        )

    def compile(
        self,
        package: ModelPackage,
        device: DeviceInfo,
        options: CompileOptions,
    ) -> ArtifactVariant:
        del package, device, options
        raise UnsupportedBackendError(
            "Horizon compile support is a declared extension point, not implemented"
        )

    def load(self, artifact: ArtifactVariant) -> BackendSession:
        del artifact
        raise UnsupportedBackendError(
            "Horizon load support is a declared extension point, not implemented"
        )
