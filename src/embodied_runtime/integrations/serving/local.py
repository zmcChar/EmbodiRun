"""Generic local provider composed from a model adapter and hardware backend."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from embodied_runtime.backends import BackendRegistry
from embodied_runtime.contracts import (
    CompileOptions,
    DeviceInfo,
    InferenceRequest,
    InferenceResult,
    ModelAdapter,
    ModelPackage,
    RawRequest,
)
from embodied_runtime.engine import EngineConfig, ExecutionEngine

from .base import ProviderCapabilities


def _candidate_devices(
    registry: BackendRegistry,
    *,
    backend_name: str | None,
    device: str | DeviceInfo | None,
) -> tuple[DeviceInfo, ...]:
    if isinstance(device, DeviceInfo):
        if backend_name is not None and device.backend != backend_name:
            raise ValueError(
                f"device belongs to backend {device.backend!r}, expected {backend_name!r}"
            )
        return (device,)

    devices = registry.probe()
    filtered = tuple(
        candidate
        for candidate in devices
        if (backend_name is None or candidate.backend == backend_name)
        and (device is None or candidate.device_id == device)
    )
    if filtered:
        return filtered

    detected = (
        ", ".join(f"{candidate.backend}/{candidate.device_id}" for candidate in devices) or "<none>"
    )
    requested_backend = "*" if backend_name is None else backend_name
    requested_device = "*" if device is None else device
    raise RuntimeError(
        "no local device matches "
        f"backend={requested_backend!r}, device={requested_device!r}; "
        f"detected devices: {detected}"
    )


class LocalBackendProvider:
    """Expose any fourth-group ``Backend`` through the provider interface.

    The provider owns only composition and request normalization.  Scheduling,
    batching, memory policy, and device execution remain delegated to the
    existing ``ExecutionEngine`` and ``BackendSession``.
    """

    provider_name = "local"
    provider_runtime = "local_backend"

    def __init__(
        self,
        adapter: ModelAdapter,
        engine: ExecutionEngine,
        *,
        capabilities: ProviderCapabilities | None = None,
        provider_name: str | None = None,
        provider_runtime: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.engine = engine
        self.provider_name = provider_name or type(self).provider_name
        self.provider_runtime = provider_runtime or type(self).provider_runtime
        self._closed = False

        if capabilities is None:
            session = getattr(engine, "session", None)
            device = getattr(session, "device", None)
            capabilities = ProviderCapabilities(
                name=self.provider_name,
                runtime=self.provider_runtime,
                model=adapter.describe(),
                is_remote=False,
                device=device if isinstance(device, DeviceInfo) else None,
                features=frozenset({"async_infer", "local_backend"}),
            )
        self._capabilities = capabilities

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    @classmethod
    def from_package(
        cls,
        adapter: ModelAdapter,
        package: ModelPackage,
        *,
        backends: BackendRegistry,
        device: str | DeviceInfo | None = None,
        backend_name: str | None = None,
        compile_options: CompileOptions | None = None,
        engine_config: EngineConfig | None = None,
        provider_name: str | None = None,
        provider_runtime: str | None = None,
    ) -> LocalBackendProvider:
        candidates = _candidate_devices(
            backends,
            backend_name=backend_name,
            device=device,
        )
        backend, selected, support = backends.select(package, candidates)
        artifact = backend.compile(
            package,
            selected,
            compile_options or CompileOptions(),
        )
        session = backend.load(artifact)
        try:
            engine = ExecutionEngine(
                package,
                session,
                config=engine_config,
                batcher=adapter.collate,
                splitter=adapter.unbatch,
            )
        except Exception:
            session.close()
            raise

        resolved_name = provider_name or cls.provider_name
        resolved_runtime = provider_runtime or cls.provider_runtime
        capabilities = ProviderCapabilities(
            name=resolved_name,
            runtime=resolved_runtime,
            model=package.spec,
            is_remote=False,
            device=selected,
            features=frozenset(
                {
                    "async_infer",
                    "local_backend",
                    *support.capabilities,
                }
            ),
        )
        return cls(
            adapter,
            engine,
            capabilities=capabilities,
            provider_name=resolved_name,
            provider_runtime=resolved_runtime,
        )

    @classmethod
    def from_checkpoint(
        cls,
        adapter: ModelAdapter,
        checkpoint: str,
        *,
        backends: BackendRegistry,
        device: str | DeviceInfo | None = None,
        backend_name: str | None = None,
        compile_options: CompileOptions | None = None,
        engine_config: EngineConfig | None = None,
        package_options: dict[str, Any] | None = None,
        provider_name: str | None = None,
        provider_runtime: str | None = None,
    ) -> LocalBackendProvider:
        package = adapter.build_package(checkpoint, **(package_options or {}))
        return cls.from_package(
            adapter,
            package,
            backends=backends,
            device=device,
            backend_name=backend_name,
            compile_options=compile_options,
            engine_config=engine_config,
            provider_name=provider_name,
            provider_runtime=provider_runtime,
        )

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")

        payload = request.payload
        if isinstance(payload, RawRequest):
            payload = self.adapter.preprocess_one(payload)
        prepared = replace(request, payload=payload)
        result = await self.engine.infer_async(prepared)
        return InferenceResult(
            request_id=result.request_id,
            output=result.output,
            status=result.status,
            queue_time_s=result.queue_time_s,
            execution_time_s=result.execution_time_s,
            metadata={
                **result.metadata,
                "provider": self.provider_name,
                "provider_runtime": self.provider_runtime,
            },
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.engine.aclose()
        self._closed = True


__all__ = ["LocalBackendProvider"]
