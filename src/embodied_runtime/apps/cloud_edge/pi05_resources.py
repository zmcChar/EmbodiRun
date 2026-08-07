"""Heavy model/backend resource construction for pi0.5 collaboration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .pi05_settings import Pi05CpuGpuConfig


@dataclass(slots=True)
class Pi05CollaborationResources:
    cloud_adapter: Any
    cloud_engine: Any
    cloud_session: Any
    edge_adapter: Any
    edge_engine: Any
    edge_session: Any
    load_time_s: float
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.edge_engine.close()
        finally:
            self.cloud_engine.close()


def build_pi05_collaboration_resources(
    config: Pi05CpuGpuConfig,
) -> Pi05CollaborationResources:
    """Build both placed engines; optional model dependencies load only here."""

    from embodied_runtime.backends.compile import CompileOptions
    from embodied_runtime.backends.torch_cuda import TorchCudaBackend
    from embodied_runtime.engine import ExecutionEngine
    from embodied_runtime.models.vla.pi05 import Pi05Adapter
    from embodied_runtime.models.vla.toy_single_forward import ToySingleForwardAdapter

    backend = TorchCudaBackend()
    devices = {device.device_id: device for device in backend.probe()}
    try:
        cloud_device = devices[config.cloud_device]
        edge_device = devices[config.edge_device]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"requested device {error.args[0]!r} is unavailable; detected: {available}"
        ) from error

    load_started = time.perf_counter()
    cloud_adapter = Pi05Adapter()
    cloud_package = cloud_adapter.build_package(
        config.checkpoint,
        local_files_only=config.local_files_only,
    )
    cloud_options: dict[str, Any] = {"empty_cache_on_close": True}
    if config.cuda_graph:
        cloud_options["cuda_graph_entrypoints"] = (cloud_package.plan.step,)
        cloud_options["cuda_graph_dynamic_scalar_inputs"] = {
            cloud_package.plan.step: ("time",),
        }
    cloud_artifact = backend.compile(
        cloud_package,
        cloud_device,
        CompileOptions(
            mode="eager",
            dtype=None if config.dtype == "preserve" else config.dtype,
            options=cloud_options,
        ),
    )
    cloud_session = backend.load(cloud_artifact)
    try:
        cloud_engine = ExecutionEngine(
            cloud_package,
            cloud_session,
            batcher=cloud_adapter.collate,
            splitter=cloud_adapter.unbatch,
        )
    except Exception:
        cloud_session.close()
        raise

    edge_adapter = ToySingleForwardAdapter(
        action_horizon=int(cloud_package.spec.action_horizon or 0),
        action_dim=int(cloud_package.spec.action_dim or 0),
    )
    try:
        edge_package = edge_adapter.build_package()
        edge_artifact = backend.compile(
            edge_package,
            edge_device,
            CompileOptions(mode="eager", dtype="float32"),
        )
        edge_session = backend.load(edge_artifact)
        try:
            edge_engine = ExecutionEngine(
                edge_package,
                edge_session,
                batcher=edge_adapter.collate,
                splitter=edge_adapter.unbatch,
            )
        except Exception:
            edge_session.close()
            raise
    except Exception:
        cloud_engine.close()
        raise

    return Pi05CollaborationResources(
        cloud_adapter=cloud_adapter,
        cloud_engine=cloud_engine,
        cloud_session=cloud_session,
        edge_adapter=edge_adapter,
        edge_engine=edge_engine,
        edge_session=edge_session,
        load_time_s=time.perf_counter() - load_started,
    )
