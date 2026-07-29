"""Run SmolVLA at the edge with an asynchronous remote pi0.5 authority.

The edge model follows the same formal path as the local pi0.5 application:

``SmolVLAAdapter -> TorchCudaBackend -> ExecutionEngine``.

The remote pi0.5 result is deliberately kept in its native action space.
SmolVLA and pi0.5 currently describe different robot action contracts, so this
application supports source selection and failover but never numeric blending.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import (
    CompileOptions,
    InferenceRequest,
    InferenceResult,
    IterativeFlowPlan,
)
from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverDecision,
    FailoverMode,
)
from embodied_runtime.distributed.communication import TcpJsonRequestClient
from embodied_runtime.engine import ExecutionEngine

_PI05_ACTION_HORIZON = 50
_PI05_ACTION_DIM = 32
_SUPPORTED_MODES = frozenset((FailoverMode.ASYNC_CLOUD_PREFERRED, FailoverMode.EDGE_ONLY))


@dataclass(frozen=True, slots=True)
class SmolVLAPi05AsyncConfig:
    """Placement and coordination settings for the real two-host prototype."""

    edge_checkpoint: str
    edge_vlm_base_path: str
    cloud_host: str
    cloud_port: int = 18765
    edge_device: str = "cuda:0"
    stats_variant: str = "so100"
    num_steps: int = 10
    language_length: int = 48
    seed: int = 0
    local_files_only: bool = True
    tcp_timeout_s: float = 120.0
    failover: FailoverConfig = field(
        default_factory=lambda: FailoverConfig(
            mode=FailoverMode.ASYNC_CLOUD_PREFERRED,
            cloud_request_timeout_s=120.0,
            cloud_result_ttl_s=120.0,
            max_cloud_sequence_lag=4,
            # Keep the completed result cached for the second demo tick.
            cloud_submit_interval_s=60.0,
        )
    )

    def __post_init__(self) -> None:
        if not self.edge_checkpoint:
            raise ValueError("edge_checkpoint must not be empty")
        if not self.edge_vlm_base_path:
            raise ValueError("edge_vlm_base_path must not be empty")
        if not self.cloud_host:
            raise ValueError("cloud_host must not be empty")
        if not 0 < self.cloud_port < 65536:
            raise ValueError("cloud_port must be between 1 and 65535")
        if not self.edge_device.startswith("cuda:"):
            raise ValueError("edge_device must be cuda:N")
        if not self.stats_variant:
            raise ValueError("stats_variant must not be empty")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if self.language_length <= 0:
            raise ValueError("language_length must be greater than zero")
        if not math.isfinite(self.tcp_timeout_s) or self.tcp_timeout_s <= 0:
            raise ValueError("tcp_timeout_s must be greater than zero")
        if self.failover.mode not in _SUPPORTED_MODES:
            raise ValueError(
                "SmolVLA/pi0.5 supports only async_cloud_preferred or edge_only; "
                "async_blend is invalid because the action semantics differ"
            )


@dataclass(slots=True)
class _EdgeRuntime:
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
        return _pi05_response_to_result(response, request.request_id)


def run_smolvla_pi05_async(
    config: SmolVLAPi05AsyncConfig,
    *,
    edge_runtime_factory: Callable[[SmolVLAPi05AsyncConfig], _EdgeRuntime] | None = None,
    cloud_endpoint_factory: Callable[[SmolVLAPi05AsyncConfig], Pi05TcpEndpoint] | None = None,
) -> dict[str, Any]:
    """Build the edge model once and execute the three-tick async scenario."""

    build_edge = edge_runtime_factory or _build_edge_runtime
    build_cloud = cloud_endpoint_factory or _build_cloud_endpoint
    resources = build_edge(config)
    try:
        cloud = build_cloud(config)
        return asyncio.run(_run_async_scenario(resources, cloud, config))
    except BaseException:
        resources.close()
        raise


def _build_edge_runtime(
    config: SmolVLAPi05AsyncConfig,
    *,
    adapter_factory: Callable[[], Any] | None = None,
    backend_factory: Callable[[], Any] = TorchCudaBackend,
    engine_factory: Callable[..., Any] = ExecutionEngine,
) -> _EdgeRuntime:
    """Construct one persistent Adapter -> Backend -> Engine edge runtime."""

    if adapter_factory is None:
        # SmolVLA's optional LeRobot dependency remains lazy for core installs.
        from embodied_runtime.models.vla.smolvla import SmolVLAAdapter

        adapter_factory = SmolVLAAdapter

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

    return _EdgeRuntime(
        adapter=adapter,
        engine=engine,
        session=session,
        payload=payload,
        load_time_s=time.perf_counter() - load_started,
    )


def _build_cloud_endpoint(config: SmolVLAPi05AsyncConfig) -> Pi05TcpEndpoint:
    client = TcpJsonRequestClient(
        config.cloud_host,
        config.cloud_port,
        timeout_s=config.tcp_timeout_s,
    )
    return Pi05TcpEndpoint(client)


async def _run_async_scenario(
    resources: _EdgeRuntime,
    cloud: Pi05TcpEndpoint,
    config: SmolVLAPi05AsyncConfig,
) -> dict[str, Any]:
    coordinator = AsyncFailoverCoordinator(
        edge=resources.engine,
        cloud=cloud,
        config=config.failover,
    )
    try:
        await resources.engine.start()

        first, first_time_s = await _timed_tick(
            coordinator,
            resources,
            config,
            tick=1,
        )
        cloud_in_flight_after_first = coordinator.cloud_request_in_flight

        cloud_wait_started = time.perf_counter()
        if config.failover.mode is not FailoverMode.EDGE_ONLY:
            # This wait is deliberately outside the control-tick critical path.
            await asyncio.wait_for(
                coordinator.wait_for_cloud_idle(),
                timeout=config.failover.cloud_request_timeout_s + 5.0,
            )
        outside_control_cloud_wait_time_s = time.perf_counter() - cloud_wait_started

        second, second_time_s = await _timed_tick(
            coordinator,
            resources,
            config,
            tick=2,
        )

        # Logical disconnection changes the endpoint epoch, immediately
        # invalidating any cached cloud authority without changing OS networking.
        cloud.set_connected(False)
        disconnected, disconnected_time_s = await _timed_tick(
            coordinator,
            resources,
            config,
            tick=3,
        )

        edge_spec = resources.engine.package.spec
        return {
            "status": "ok",
            "mode": config.failover.mode.value,
            "edge_model_id": edge_spec.model_id,
            "edge_device": resources.session.device.device_id,
            "edge_dtype_policy": "preserve",
            "edge_load_time_s": resources.load_time_s,
            "cloud_transport": "length_prefixed_json_v1",
            "cloud_host": config.cloud_host,
            "cloud_port": config.cloud_port,
            "cloud_in_flight_after_first": cloud_in_flight_after_first,
            "outside_control_cloud_wait_time_s": outside_control_cloud_wait_time_s,
            "first_tick": _decision_record(first, first_time_s),
            "second_tick": _decision_record(second, second_time_s),
            "disconnected_tick": _decision_record(
                disconnected,
                disconnected_time_s,
            ),
            "action_contracts": {
                "edge_smolvla": {
                    "shape": [edge_spec.action_horizon, edge_spec.action_dim],
                    "semantics": "smolvla_robot_specific_action_space",
                },
                "cloud_pi05": {
                    "shape": [_PI05_ACTION_HORIZON, _PI05_ACTION_DIM],
                    "semantics": "pi05_policy_action_space",
                },
                "semantically_compatible": False,
            },
            "numeric_blend_performed": False,
            "input_mode": "independent_synthetic_payloads",
            "control_path_contract": (
                "each tick awaits edge inference only; cloud completion is consumed "
                "opportunistically or awaited outside the control path"
            ),
        }
    finally:
        await coordinator.aclose()
        await resources.engine.aclose()
        resources._closed = True


async def _timed_tick(
    coordinator: AsyncFailoverCoordinator,
    resources: _EdgeRuntime,
    config: SmolVLAPi05AsyncConfig,
    *,
    tick: int,
) -> tuple[FailoverDecision[InferenceResult], float]:
    started = time.perf_counter()
    decision = await coordinator.infer_async(
        InferenceRequest(
            payload=resources.payload,
            request_id=f"smolvla-edge-{tick}",
            num_steps=config.num_steps,
            seed=config.seed + tick - 1,
        ),
        cloud_request=InferenceRequest(
            payload={},
            request_id=f"pi05-cloud-{tick}",
            num_steps=config.num_steps,
            seed=config.seed + tick - 1,
        ),
    )
    return decision, time.perf_counter() - started


def _decision_record(
    decision: FailoverDecision[InferenceResult],
    control_path_time_s: float,
) -> dict[str, Any]:
    return {
        "source": decision.source.value,
        "sequence_id": decision.sequence_id,
        "source_sequence_id": decision.source_sequence_id,
        "connection_epoch": decision.connection_epoch,
        "fallback_reason": (
            decision.fallback_reason.value if decision.fallback_reason is not None else None
        ),
        "cloud_result_age_s": decision.cloud_result_age_s,
        "selected_action_shape": list(_action_shape(decision.result.output)),
        "selected_model_id": decision.result.metadata.get("model_id"),
        "selected_execution_time_s": decision.result.execution_time_s,
        "control_path_time_s": control_path_time_s,
    }


def _pi05_response_to_result(
    response: Mapping[str, Any],
    expected_request_id: str,
) -> InferenceResult:
    if response.get("ok") is not True:
        error_type = str(response.get("error_type") or "CloudError")
        message = str(response.get("error") or "pi0.5 cloud request failed")
        raise RuntimeError(f"{error_type}: {message}")
    if response.get("kind") != "inference_result":
        raise RuntimeError(f"unexpected cloud response kind: {response.get('kind')!r}")
    if response.get("request_id") != expected_request_id:
        raise RuntimeError("pi0.5 cloud response request_id does not match")

    actions = response.get("actions")
    observed_shape = _action_shape(actions)
    if observed_shape == (1, _PI05_ACTION_HORIZON, _PI05_ACTION_DIM):
        actions = actions[0]
        observed_shape = _action_shape(actions)
    if observed_shape != (_PI05_ACTION_HORIZON, _PI05_ACTION_DIM):
        raise RuntimeError(
            "pi0.5 cloud actions must have shape "
            f"[{_PI05_ACTION_HORIZON}, {_PI05_ACTION_DIM}], got {observed_shape}"
        )
    _validate_finite_numbers(actions)

    reported_shape = response.get("action_shape")
    if reported_shape not in (
        None,
        [_PI05_ACTION_HORIZON, _PI05_ACTION_DIM],
        [1, _PI05_ACTION_HORIZON, _PI05_ACTION_DIM],
    ):
        raise RuntimeError(
            f"pi0.5 cloud action_shape disagrees with its actions: {reported_shape!r}"
        )
    execution_time_s = float(response.get("execution_time_s") or 0.0)
    if not math.isfinite(execution_time_s) or execution_time_s < 0:
        raise RuntimeError("pi0.5 cloud execution_time_s must be finite and non-negative")

    return InferenceResult(
        request_id=expected_request_id,
        output={"actions": actions},
        execution_time_s=execution_time_s,
        metadata={
            "model_id": response.get("model_id"),
            "device_id": response.get("device"),
            "action_shape": observed_shape,
            "action_space_id": "pi05_policy_action_space",
        },
    )


def _actions(output: Any) -> Any:
    return output["actions"] if isinstance(output, Mapping) else output


def _action_shape(output: Any) -> tuple[int, ...]:
    actions = _actions(output)
    shape = getattr(actions, "shape", None)
    if shape is not None:
        return tuple(int(dimension) for dimension in shape)
    return _sequence_shape(actions)


def _sequence_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    length = len(value)
    if length == 0:
        return (0,)
    child_shape = _sequence_shape(value[0])
    if any(_sequence_shape(child) != child_shape for child in value[1:]):
        raise RuntimeError("action values must form a rectangular tensor")
    return (length, *child_shape)


def _validate_finite_numbers(value: Any) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_numbers(item)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("pi0.5 cloud actions must contain only numbers")
    if not math.isfinite(float(value)):
        raise RuntimeError("pi0.5 cloud actions contain a non-finite value")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real SmolVLA through Adapter/Backend/Engine on an edge GPU and "
            "use an asynchronous remote pi0.5 result when available."
        )
    )
    parser.add_argument("--edge-checkpoint", required=True)
    parser.add_argument("--edge-vlm-base-path", required=True)
    parser.add_argument("--cloud-host", required=True)
    parser.add_argument("--cloud-port", type=int, default=18765)
    parser.add_argument("--edge-device", default="cuda:0")
    parser.add_argument("--stats-variant", default="so100")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--language-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tcp-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--mode",
        choices=(
            FailoverMode.ASYNC_CLOUD_PREFERRED.value,
            FailoverMode.EDGE_ONLY.value,
        ),
        default=FailoverMode.ASYNC_CLOUD_PREFERRED.value,
    )
    parser.add_argument("--cloud-result-ttl-s", type=float, default=120.0)
    parser.add_argument("--max-cloud-sequence-lag", type=int, default=4)
    parser.add_argument("--cloud-submit-interval-s", type=float, default=60.0)
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = SmolVLAPi05AsyncConfig(
        edge_checkpoint=args.edge_checkpoint,
        edge_vlm_base_path=args.edge_vlm_base_path,
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        edge_device=args.edge_device,
        stats_variant=args.stats_variant,
        num_steps=args.num_steps,
        language_length=args.language_length,
        seed=args.seed,
        local_files_only=not args.allow_download,
        tcp_timeout_s=args.tcp_timeout_s,
        failover=FailoverConfig(
            mode=FailoverMode(args.mode),
            cloud_request_timeout_s=args.tcp_timeout_s,
            cloud_result_ttl_s=args.cloud_result_ttl_s,
            max_cloud_sequence_lag=args.max_cloud_sequence_lag,
            cloud_submit_interval_s=args.cloud_submit_interval_s,
        ),
    )
    summary = run_smolvla_pi05_async(config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
