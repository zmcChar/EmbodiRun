"""Runtime construction and TCP endpoint for SmolVLA/pi0.5 placement."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from embodied_runtime.distributed.communication import TcpJsonRequestClient
from embodied_runtime.engine.request import InferenceRequest
from embodied_runtime.engine.result import InferenceResult

from .pi05_codec import pi05_response_to_result
from .smolvla_settings import SmolVLAPi05AsyncConfig


@dataclass(slots=True)
class EdgeRuntime:
    adapter: Any
    engine: Any
    session: Any
    payload: Mapping[str, Any]
    load_time_s: float
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.engine.close()
            self._closed = True


class Pi05TcpEndpoint:
    """Adapt the existing TCP/JSON client to the failover endpoint contract."""

    def __init__(self, client: TcpJsonRequestClient) -> None:
        self.client = client
        self._connected = True
        self._connection_epoch = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def connection_epoch(self) -> int:
        return self._connection_epoch

    def set_connected(self, connected: bool) -> None:
        """Change logical connectivity without embedding host-specific controls."""

        if connected != self._connected:
            self._connected = connected
            self._connection_epoch += 1

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        epoch = self.connection_epoch
        if not self.connected:
            raise ConnectionError("pi0.5 cloud endpoint is logically disconnected")
        response = await self.client.request(
            {
                "kind": "infer",
                "request_id": request.request_id,
                "num_steps": request.num_steps,
                "seed": request.seed,
            }
        )
        if not self.connected or epoch != self.connection_epoch:
            raise ConnectionError("pi0.5 cloud connection changed during inference")
        return pi05_response_to_result(response, request.request_id)


def build_edge_runtime(
    config: SmolVLAPi05AsyncConfig,
    *,
    adapter_factory: Callable[[], Any] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> EdgeRuntime:
    """Construct one persistent Adapter -> Backend -> Engine edge runtime."""

    # Heavy model/backend dependencies stay behind the actual build path.
    from embodied_runtime.backends.compile import CompileOptions
    from embodied_runtime.models.plans import IterativeFlowPlan

    if adapter_factory is None:
        from embodied_runtime.models.vla.smolvla import SmolVLAAdapter

        adapter_factory = SmolVLAAdapter
    if backend_factory is None:
        from embodied_runtime.backends.torch_cuda import TorchCudaBackend

        backend_factory = TorchCudaBackend
    if engine_factory is None:
        from embodied_runtime.engine import ExecutionEngine

        engine_factory = ExecutionEngine

    load_started = time.perf_counter()
    adapter = adapter_factory()
    package = adapter.build_package(
        config.edge_checkpoint,
        vlm_base_path=config.edge_vlm_base_path,
        stats_variant=config.stats_variant,
        local_files_only=config.local_files_only,
    )
    if not isinstance(package.plan, IterativeFlowPlan):
        raise TypeError("SmolVLA adapter must expose an IterativeFlowPlan")

    backend = backend_factory()
    devices = {device.device_id: device for device in backend.probe()}
    try:
        device = devices[config.edge_device]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"edge device {config.edge_device!r} is unavailable; detected: {available}"
        ) from error

    artifact = backend.compile(
        package,
        device,
        CompileOptions(
            mode="eager",
            # Preserve the checkpoint's BF16 VLM and FP32 flow/action modules.
            dtype=None,
            options={"empty_cache_on_close": True},
        ),
    )
    session = backend.load(artifact)
    try:
        engine = engine_factory(
            package,
            session,
            batcher=adapter.collate,
            splitter=adapter.unbatch,
        )
    except Exception:
        session.close()
        raise
    try:
        payload = adapter.synthetic_batch(
            batch_size=1,
            language_length=config.language_length,
            seed=config.seed,
        )
    except Exception:
        engine.close()
        raise

    return EdgeRuntime(
        adapter=adapter,
        engine=engine,
        session=session,
        payload=payload,
        load_time_s=time.perf_counter() - load_started,
    )


def build_cloud_endpoint(config: SmolVLAPi05AsyncConfig) -> Pi05TcpEndpoint:
    client = TcpJsonRequestClient(
        config.cloud_host,
        config.cloud_port,
        timeout_s=config.tcp_timeout_s,
    )
    return Pi05TcpEndpoint(client)


# Existing integration users imported this private name before the split.
_EdgeRuntime = EdgeRuntime
