"""Local NVIDIA/Transformers provider for the GR00T model contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import (
    CompileOptions,
    InferenceRequest,
    InferenceResult,
    RawRequest,
)
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_CHECKPOINT,
    Gr00tN17Adapter,
)


def _select_device(backend: TorchCudaBackend, device_id: str):
    devices = {device.device_id: device for device in backend.probe()}
    try:
        return devices[device_id]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"device {device_id!r} is unavailable; detected devices: {available}"
        ) from error


def _prepared_request(
    adapter: Gr00tN17Adapter,
    request: InferenceRequest,
) -> InferenceRequest:
    if not isinstance(request, InferenceRequest):
        raise TypeError("request must be an InferenceRequest")
    payload = request.payload
    if isinstance(payload, RawRequest):
        payload = adapter.preprocess_one(payload)
    elif not isinstance(payload, Mapping):
        raise TypeError("GR00T provider payload must be RawRequest or a mapping")
    return replace(request, payload=payload)


class HfLocalGr00tProvider:
    """NVIDIA's reference GR00T policy through the local formal runtime."""

    provider_name = "hf"

    def __init__(
        self,
        adapter: Gr00tN17Adapter,
        engine: ExecutionEngine,
    ) -> None:
        self.adapter = adapter
        self.engine = engine
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str = "cuda:0",
        mode: str = "eager",
        local_files_only: bool = True,
        cache_dir: str | None = None,
        revision: str | None = None,
        strict: bool = True,
        adapter: Gr00tN17Adapter | None = None,
    ) -> HfLocalGr00tProvider:
        if mode != "eager":
            raise ValueError(
                "GR00T N1.7 currently supports only mode='eager'; compiling the "
                "full reference policy would incorrectly include NumPy preprocessing "
                "and CPU action decoding"
            )
        adapter = adapter or Gr00tN17Adapter()
        package = adapter.build_package(
            checkpoint,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            revision=revision,
            strict=strict,
        )
        backend = TorchCudaBackend()
        selected = _select_device(backend, device)
        artifact = backend.compile(
            package,
            selected,
            CompileOptions(
                mode=mode,
                # Preserve the official policy's BF16/FP32 choices.
                dtype=None,
                options={
                    "offload_module_on_close": True,
                    "empty_cache_on_close": True,
                },
            ),
        )
        session = backend.load(artifact)
        try:
            engine = ExecutionEngine(
                package,
                session,
                batcher=adapter.collate,
                splitter=adapter.unbatch,
            )
        except Exception:
            session.close()
            raise
        return cls(adapter, engine)

    @property
    def connected(self) -> bool:
        return not self._closed

    @property
    def connection_epoch(self) -> int:
        return 0

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        if self._closed:
            raise RuntimeError("HF GR00T provider is closed")
        prepared = _prepared_request(self.adapter, request)
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
                "provider_runtime": "nvidia_isaac_gr00t",
            },
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        await self.engine.aclose()
        self._closed = True


__all__ = ["HfLocalGr00tProvider"]
