"""Run real π0.5 on CUDA and a lightweight policy on CPU with async fusion."""

from __future__ import annotations

import argparse
import asyncio
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import CompileOptions, InferenceRequest, InferenceResult, RawRequest
from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverMode,
    ResultSource,
)
from embodied_runtime.distributed.communication import (
    DummyLink,
    DummyRemoteInferenceEndpoint,
)
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.pi05 import Pi05Adapter
from embodied_runtime.models.vla.toy_single_forward import ToySingleForwardAdapter

from .action_fusion import make_action_result_fuser


@dataclass(frozen=True, slots=True)
class Pi05CpuGpuConfig:
    checkpoint: str = "lerobot/pi05_base"
    cloud_device: str = "cuda:0"
    edge_device: str = "cpu"
    dtype: str = "preserve"
    num_steps: int = 10
    language_length: int = 8
    seed: int = 0
    cuda_graph: bool = False
    local_files_only: bool = True
    one_way_latency_s: float = 0.0
    cloud_weight: float = 0.5
    failover: FailoverConfig = FailoverConfig(
        mode=FailoverMode.ASYNC_BLEND,
        cloud_request_timeout_s=5.0,
        cloud_result_ttl_s=5.0,
        max_cloud_sequence_lag=4,
        cloud_submit_interval_s=60.0,
    )

    def __post_init__(self) -> None:
        if not self.cloud_device.startswith("cuda:"):
            raise ValueError("cloud_device must be cuda:N for the GPU collaboration demo")
        if self.edge_device != "cpu":
            raise ValueError("edge_device must be cpu for the CPU collaboration demo")
        if self.dtype not in {"preserve", "float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if self.language_length <= 0:
            raise ValueError("language_length must be greater than zero")
        if self.one_way_latency_s < 0:
            raise ValueError("one_way_latency_s cannot be negative")
        if not 0.0 <= self.cloud_weight <= 1.0:
            raise ValueError("cloud_weight must be between zero and one")


def load_pi05_cpu_gpu_config(path: str | Path) -> Pi05CpuGpuConfig:
    """Load model placement, coordination, and dummy-link settings from TOML."""

    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    model = raw.get("model", {})
    cloud = raw.get("cloud", {})
    edge = raw.get("edge", {})
    coordination = raw.get("coordination", {})
    link = raw.get("dummy_link", {})
    return Pi05CpuGpuConfig(
        checkpoint=str(model.get("checkpoint", "lerobot/pi05_base")),
        cloud_device=str(cloud.get("device", "cuda:0")),
        edge_device=str(edge.get("device", "cpu")),
        dtype=str(cloud.get("dtype", "preserve")),
        num_steps=int(cloud.get("num_steps", 10)),
        language_length=int(cloud.get("language_length", 8)),
        seed=int(cloud.get("seed", 0)),
        cuda_graph=bool(cloud.get("cuda_graph", False)),
        local_files_only=bool(model.get("local_files_only", True)),
        one_way_latency_s=float(link.get("one_way_latency_s", 0.0)),
        cloud_weight=float(coordination.get("cloud_weight", 0.5)),
        failover=FailoverConfig(
            mode=FailoverMode(coordination.get("mode", FailoverMode.ASYNC_BLEND)),
            cloud_request_timeout_s=float(coordination.get("cloud_request_timeout_s", 5.0)),
            cloud_result_ttl_s=float(coordination.get("cloud_result_ttl_s", 5.0)),
            max_cloud_sequence_lag=int(coordination.get("max_cloud_sequence_lag", 4)),
            cloud_submit_interval_s=float(coordination.get("cloud_submit_interval_s", 60.0)),
        ),
    )


def run_pi05_cpu_gpu_collaboration(config: Pi05CpuGpuConfig) -> dict[str, Any]:
    """Build both placed models and run edge, blended, and disconnected ticks."""

    resources = _build_resources(config)
    try:
        return asyncio.run(_run_collaboration(resources, config))
    except Exception:
        resources.close()
        raise


@dataclass(slots=True)
class _Resources:
    cloud_adapter: Pi05Adapter
    cloud_engine: ExecutionEngine
    cloud_session: Any
    edge_adapter: ToySingleForwardAdapter
    edge_engine: ExecutionEngine
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


def _build_resources(config: Pi05CpuGpuConfig) -> _Resources:
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

    return _Resources(
        cloud_adapter=cloud_adapter,
        cloud_engine=cloud_engine,
        cloud_session=cloud_session,
        edge_adapter=edge_adapter,
        edge_engine=edge_engine,
        edge_session=edge_session,
        load_time_s=time.perf_counter() - load_started,
    )


async def _run_collaboration(
    resources: _Resources,
    config: Pi05CpuGpuConfig,
) -> dict[str, Any]:
    torch = _torch()
    link = DummyLink(one_way_latency_s=config.one_way_latency_s)
    remote_cloud = DummyRemoteInferenceEndpoint(resources.cloud_engine, link)
    base_fuser = make_action_result_fuser(cloud_weight=config.cloud_weight)
    fusion_evidence: dict[str, Any] = {}

    def record_fusion(
        edge_result: InferenceResult,
        cloud_result: InferenceResult,
    ) -> InferenceResult:
        fused = base_fuser(edge_result, cloud_result)
        fusion_evidence["edge"] = _actions(edge_result).detach().float().cpu().clone()
        fusion_evidence["cloud"] = _actions(cloud_result).detach().float().cpu().clone()
        fusion_evidence["fused"] = _actions(fused).detach().float().cpu().clone()
        return fused

    coordinator = AsyncFailoverCoordinator(
        edge=resources.edge_engine,
        cloud=remote_cloud,
        config=config.failover,
        fuser=record_fusion,
    )
    cloud_payload = resources.cloud_adapter.synthetic_batch(
        batch_size=1,
        language_length=config.language_length,
        seed=config.seed,
    )

    try:
        await resources.edge_engine.start()
        await resources.cloud_engine.start()
        torch.cuda.reset_peak_memory_stats(torch.device(config.cloud_device))
        first_started = time.perf_counter()
        first = await coordinator.infer_async(
            _edge_request(resources.edge_adapter, tick=1),
            cloud_request=_cloud_request(
                cloud_payload,
                tick=1,
                num_steps=config.num_steps,
                seed=config.seed,
            ),
        )
        first_tick_time_s = time.perf_counter() - first_started
        cloud_in_flight_after_first = coordinator.cloud_request_in_flight

        if config.failover.mode is not FailoverMode.EDGE_ONLY:
            await asyncio.wait_for(
                coordinator.wait_for_cloud_idle(),
                timeout=config.failover.cloud_request_timeout_s + 5.0,
            )

        second = await coordinator.infer_async(
            _edge_request(resources.edge_adapter, tick=2),
            cloud_request=_cloud_request(
                cloud_payload,
                tick=2,
                num_steps=config.num_steps,
                seed=config.seed + 1,
            ),
        )

        link.set_connected(False)
        disconnected = await coordinator.infer_async(
            _edge_request(resources.edge_adapter, tick=3),
            cloud_request=_cloud_request(
                cloud_payload,
                tick=3,
                num_steps=config.num_steps,
                seed=config.seed + 2,
            ),
        )

        summary: dict[str, Any] = {
            "mode": config.failover.mode.value,
            "cloud_model_id": resources.cloud_engine.package.spec.model_id,
            "edge_model_id": resources.edge_engine.package.spec.model_id,
            "cloud_device": resources.cloud_session.device.device_id,
            "edge_device": resources.edge_session.device.device_id,
            "load_time_s": resources.load_time_s,
            "first_tick_source": first.source.value,
            "first_tick_time_s": first_tick_time_s,
            "cloud_in_flight_after_first": cloud_in_flight_after_first,
            "second_tick_source": second.source.value,
            "second_tick_cloud_sequence": second.source_sequence_id,
            "disconnected_tick_source": disconnected.source.value,
            "disconnected_reason": (
                disconnected.fallback_reason.value
                if disconnected.fallback_reason is not None
                else None
            ),
            "action_shape": tuple(_actions(second.result).shape),
            "final_action_device": str(_actions(second.result).device),
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated(
                torch.device(config.cloud_device)
            )
            / (1024**2),
            "cloud_weight": config.cloud_weight,
        }
        if second.source is ResultSource.BLENDED:
            edge_actions = fusion_evidence["edge"]
            cloud_actions = fusion_evidence["cloud"]
            fused_actions = fusion_evidence["fused"]
            expected = edge_actions + (cloud_actions - edge_actions) * config.cloud_weight
            summary.update(
                {
                    "fusion_max_abs_error": float((fused_actions - expected).abs().max().item()),
                    "fused_vs_edge_max_abs": float(
                        (fused_actions - edge_actions).abs().max().item()
                    ),
                    "fused_vs_cloud_max_abs": float(
                        (fused_actions - cloud_actions).abs().max().item()
                    ),
                }
            )
        return summary
    finally:
        await coordinator.aclose()
        await asyncio.gather(
            resources.edge_engine.aclose(),
            resources.cloud_engine.aclose(),
        )
        resources._closed = True


def _edge_request(
    adapter: ToySingleForwardAdapter,
    *,
    tick: int,
) -> InferenceRequest:
    action_dim = adapter.action_dim
    target = [tick * (index + 1) / action_dim for index in range(action_dim)]
    payload = adapter.preprocess_one(RawRequest(observation={"target": target}))
    return InferenceRequest(payload=payload, request_id=f"cpu-edge-{tick}")


def _cloud_request(
    payload: Mapping[str, Any],
    *,
    tick: int,
    num_steps: int,
    seed: int,
) -> InferenceRequest:
    return InferenceRequest(
        payload=payload,
        request_id=f"gpu-cloud-{tick}",
        num_steps=num_steps,
        seed=seed,
    )


def _actions(result: InferenceResult) -> Any:
    output = result.output
    return output["actions"] if isinstance(output, Mapping) else output


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - model extras require torch
        raise RuntimeError("the CPU/GPU collaboration demo requires PyTorch") from error
    return torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run π0.5 on CUDA and a lightweight policy on CPU with async fusion."
    )
    parser.add_argument(
        "--config",
        default="configs/pi05_cpu_gpu_collaboration.toml",
    )
    parser.add_argument("--checkpoint", help="override model.checkpoint")
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in FailoverMode),
        help="override coordination.mode",
    )
    parser.add_argument("--num-steps", type=int, help="override cloud.num_steps")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_pi05_cpu_gpu_config(args.config)
    if args.checkpoint is not None:
        config = replace(config, checkpoint=args.checkpoint)
    if args.mode is not None:
        config = replace(
            config,
            failover=replace(config.failover, mode=FailoverMode(args.mode)),
        )
    if args.num_steps is not None:
        config = replace(config, num_steps=args.num_steps)
    if args.cuda_graph:
        config = replace(config, cuda_graph=True)
    if args.allow_download:
        config = replace(config, local_files_only=False)

    summary = run_pi05_cpu_gpu_collaboration(config)
    print("π0.5 GPU + lightweight CPU collaboration succeeded")
    for name, value in summary.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
