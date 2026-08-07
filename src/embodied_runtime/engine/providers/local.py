"""Local inference provider composed from a model adapter and hardware backend."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import replace
from typing import Any

from embodied_runtime.backends import BackendRegistry
from embodied_runtime.backends.compile import CompileOptions
from embodied_runtime.backends.device import DeviceInfo
from embodied_runtime.engine.config import EngineConfig
from embodied_runtime.engine.execution_engine import ExecutionEngine
from embodied_runtime.engine.provider import ProviderCapabilities
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult
from embodied_runtime.models.interfaces import ModelAdapter
from embodied_runtime.models.package import ModelPackage
from embodied_runtime.models.request import RawRequest


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
    """Expose any model-agnostic ``Backend`` through the provider interface.

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
        self._resources_closed = False
        self._close_lock = asyncio.Lock()
        self._preprocess_tasks: set[asyncio.Task[Any]] = set()

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
        provider_features: frozenset[str] = frozenset(),
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
                    *provider_features,
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
        provider_features: frozenset[str] = frozenset(),
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
            provider_features=provider_features,
        )

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if self._closed:
            raise RuntimeError(f"{self.provider_name} provider is closed")
        if not isinstance(request, InferenceRequest):
            raise TypeError("request must be an InferenceRequest")

        payload = request.payload
        if isinstance(payload, RawRequest):
            preprocess = asyncio.create_task(
                asyncio.to_thread(self.adapter.preprocess_one, payload),
                name=f"preprocess-{request.request_id}",
            )
            self._preprocess_tasks.add(preprocess)

            def forget_preprocess(completed: asyncio.Task[Any]) -> None:
                self._preprocess_tasks.discard(completed)
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    completed.result()

            preprocess.add_done_callback(forget_preprocess)
            # Cancellation stops this request but cannot stop its worker
            # thread. Shield it so aclose() can drain preprocessing before the
            # adapter/backend resources are released.
            try:
                payload = await asyncio.shield(preprocess)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await preprocess
                raise
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
        async with self._close_lock:
            if self._resources_closed:
                return
            self._closed = True
            preprocessing = tuple(self._preprocess_tasks)
            if preprocessing:
                await asyncio.gather(*preprocessing, return_exceptions=True)
            await self.engine.aclose()
            self._resources_closed = True


__all__ = ["LocalBackendProvider"]
