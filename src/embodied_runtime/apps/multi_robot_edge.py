"""Run one robot-scoped edge model against a shared cloud model service."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import hashlib
import json
import math
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from embodied_runtime.contracts import InferenceRequest, InferenceResult, RawRequest
from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverDecision,
    FailoverMode,
    ResultSource,
)
from embodied_runtime.distributed.communication import (
    CloudSessionProtocolError,
    CloudSessionRemoteError,
    MultiTenantTcpEndpoint,
    TcpJsonProtocolError,
    TcpJsonRequestClient,
    json_compatible,
)
from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.integrations.serving import InferenceProvider

from ._local_provider import LocalProviderConfig, build_local_provider

_SUPPORTED_MODES = frozenset((FailoverMode.ASYNC_CLOUD_PREFERRED, FailoverMode.EDGE_ONLY))
_OBSERVATION_MODES = frozenset(("target_vector", "adapter_synthetic"))


@dataclass(frozen=True, slots=True)
class MultiRobotEdgeConfig:
    identity: RobotSessionIdentity
    provider: LocalProviderConfig = field(
        default_factory=lambda: LocalProviderConfig(
            provider_name="edge-local",
        )
    )
    cloud_host: str = "localhost"
    cloud_port: int = 18770
    cloud_timeout_s: float = 5.0
    registration_timeout_s: float = 2.0
    reconnect_interval_s: float = 1.0
    physical_host_id: str = "edge-host"
    physical_resource_id: str = "local-device-0"
    failover: FailoverConfig = field(
        default_factory=lambda: FailoverConfig(
            mode=FailoverMode.ASYNC_CLOUD_PREFERRED,
            cloud_request_timeout_s=5.0,
            cloud_result_ttl_s=2.0,
            max_cloud_sequence_lag=2,
            cloud_submit_interval_s=0.0,
        )
    )
    ticks: int = 8
    control_period_s: float = 0.05
    observation_mode: str = "target_vector"
    observation_offset: float = 0.0
    observation_seed: int = 0
    observation_language_length: int = 48
    disconnect_tick: int | None = None
    reconnect_tick: int | None = None

    def __post_init__(self) -> None:
        if not self.cloud_host:
            raise ValueError("cloud_host must not be empty")
        if not 0 < self.cloud_port < 65536:
            raise ValueError("cloud_port must be between 1 and 65535")
        if not math.isfinite(self.cloud_timeout_s) or self.cloud_timeout_s <= 0:
            raise ValueError("cloud_timeout_s must be finite and greater than zero")
        if (
            not math.isfinite(self.registration_timeout_s)
            or self.registration_timeout_s <= 0
        ):
            raise ValueError("registration_timeout_s must be finite and greater than zero")
        if not math.isfinite(self.reconnect_interval_s) or self.reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s must be finite and non-negative")
        if not self.physical_host_id.strip():
            raise ValueError("physical_host_id must not be empty")
        if not self.physical_resource_id.strip():
            raise ValueError("physical_resource_id must not be empty")
        if self.failover.mode not in _SUPPORTED_MODES:
            raise ValueError(
                "multi-robot edge supports async_cloud_preferred or edge_only; "
                "numeric blending requires an explicit shared action contract"
            )
        if self.ticks <= 0:
            raise ValueError("ticks must be greater than zero")
        if not math.isfinite(self.control_period_s) or self.control_period_s < 0:
            raise ValueError("control_period_s must be finite and non-negative")
        if self.observation_mode not in _OBSERVATION_MODES:
            supported = ", ".join(sorted(_OBSERVATION_MODES))
            raise ValueError(f"observation_mode must be one of: {supported}")
        if not math.isfinite(self.observation_offset):
            raise ValueError("observation_offset must be finite")
        if not isinstance(self.observation_seed, int) or isinstance(
            self.observation_seed,
            bool,
        ):
            raise TypeError("observation_seed must be an integer")
        if (
            not isinstance(self.observation_language_length, int)
            or isinstance(self.observation_language_length, bool)
            or self.observation_language_length <= 0
        ):
            raise ValueError("observation_language_length must be a positive integer")


def load_multi_robot_edge_config(path: str | Path) -> MultiRobotEdgeConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    identity = raw.get("identity", {})
    provider = raw.get("edge_provider", {})
    cloud = raw.get("cloud", {})
    coordination = raw.get("coordination", {})
    demo = raw.get("demo", {})
    observation = raw.get("observation", {})
    for name, values in (
        ("identity", identity),
        ("edge_provider", provider),
        ("cloud", cloud),
        ("coordination", coordination),
        ("demo", demo),
        ("observation", observation),
    ):
        if not isinstance(values, Mapping):
            raise TypeError(f"{name} must be a TOML table")

    cloud_timeout_s = float(cloud.get("timeout_s", 5.0))
    return MultiRobotEdgeConfig(
        identity=RobotSessionIdentity(
            robot_id=str(identity.get("robot_id", "robot-demo")),
            edge_node_id=str(identity.get("edge_node_id", "edge-demo")),
            session_id=str(identity.get("session_id", "session-demo")),
            embodiment=str(identity.get("embodiment", "toy-vector")),
            action_space_id=str(identity.get("action_space_id", "toy-vector-actions-v1")),
        ),
        provider=LocalProviderConfig.from_mapping(provider),
        cloud_host=str(cloud.get("host", "localhost")),
        cloud_port=int(cloud.get("port", 18770)),
        cloud_timeout_s=cloud_timeout_s,
        registration_timeout_s=float(cloud.get("registration_timeout_s", 2.0)),
        reconnect_interval_s=float(cloud.get("reconnect_interval_s", 1.0)),
        physical_host_id=str(identity.get("physical_host_id", "edge-host")),
        physical_resource_id=str(identity.get("physical_resource_id", "local-device-0")),
        failover=FailoverConfig(
            mode=FailoverMode(
                coordination.get(
                    "mode",
                    FailoverMode.ASYNC_CLOUD_PREFERRED.value,
                )
            ),
            cloud_request_timeout_s=float(
                coordination.get("cloud_request_timeout_s", cloud_timeout_s)
            ),
            cloud_result_ttl_s=float(coordination.get("cloud_result_ttl_s", 2.0)),
            max_cloud_sequence_lag=int(coordination.get("max_cloud_sequence_lag", 2)),
            cloud_submit_interval_s=float(coordination.get("cloud_submit_interval_s", 0.0)),
        ),
        ticks=int(demo.get("ticks", 8)),
        control_period_s=float(demo.get("control_period_s", 0.05)),
        observation_mode=str(observation.get("mode", "target_vector")),
        observation_offset=float(observation.get("offset", 0.0)),
        observation_seed=int(observation.get("seed", 0)),
        observation_language_length=int(observation.get("language_length", 48)),
        disconnect_tick=_optional_int(demo.get("disconnect_tick")),
        reconnect_tick=_optional_int(demo.get("reconnect_tick")),
    )


class RobotEdgeRuntime:
    """One edge Provider, one cloud session, and one failover state machine."""

    def __init__(
        self,
        *,
        identity: RobotSessionIdentity,
        edge: InferenceProvider,
        cloud: MultiTenantTcpEndpoint,
        failover: FailoverConfig,
        registration_timeout_s: float = 2.0,
        reconnect_interval_s: float = 1.0,
    ) -> None:
        if cloud.identity != identity:
            raise ValueError("cloud endpoint identity must match edge runtime")
        if not math.isfinite(reconnect_interval_s) or reconnect_interval_s < 0:
            raise ValueError("reconnect_interval_s must be finite and non-negative")
        if not math.isfinite(registration_timeout_s) or registration_timeout_s <= 0:
            raise ValueError("registration_timeout_s must be finite and greater than zero")
        self.identity = identity
        self.edge = edge
        self.cloud = cloud
        self.failover = AsyncFailoverCoordinator(
            edge=edge,
            cloud=cloud,
            config=failover,
        )
        self.registration_timeout_s = registration_timeout_s
        self.reconnect_interval_s = reconnect_interval_s
        self._reconnect_task: asyncio.Task[Mapping[str, Any]] | None = None
        self._last_reconnect_attempt_s: float | None = None
        self._last_reconnect_error: BaseException | None = None
        self._last_sequence_id = 0
        self._inference_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._resources_closed = False

    async def start(self) -> None:
        """Attempt initial registration without making cloud availability mandatory."""

        if self._closed:
            raise RuntimeError("edge runtime is closed")
        try:
            await self._connect_with_timeout()
        except (
            CloudSessionProtocolError,
            CloudSessionRemoteError,
            ConnectionError,
            OSError,
            asyncio.TimeoutError,
            asyncio.IncompleteReadError,
            TcpJsonProtocolError,
            ValueError,
        ) as error:
            # The first control tick still has a valid local Provider.
            self._last_reconnect_error = error
            return
        self._last_reconnect_error = None

    @property
    def last_reconnect_error(self) -> BaseException | None:
        return self._last_reconnect_error

    def set_cloud_enabled(self, enabled: bool) -> None:
        self.cloud.set_enabled(enabled)
        if enabled:
            self._last_reconnect_attempt_s = None
            self._schedule_reconnect()

    async def infer_observation(
        self,
        observation: Mapping[str, Any],
        *,
        sequence_id: int,
        observation_timestamp_s: float | None = None,
        prompt: str | None = None,
        owned_cloud_observation: Mapping[str, Any] | None = None,
    ) -> FailoverDecision[InferenceResult]:
        """Run one ordered control tick.

        ``owned_cloud_observation`` opts out of the defensive snapshot copy.
        Its buffers must remain immutable until the cloud request completes.
        """

        async with self._inference_lock:
            if self._closed:
                raise RuntimeError("edge runtime is closed")
            expected_sequence_id = self._last_sequence_id + 1
            if sequence_id != expected_sequence_id:
                raise ValueError(
                    "sequence_id must be contiguous starting at 1; "
                    f"expected {expected_sequence_id}, received {sequence_id}"
                )
            # A failed inference consumes the sequence by design (at-most-once).
            self._last_sequence_id = sequence_id
            self._schedule_reconnect()
            timestamp_s = (
                time.monotonic() if observation_timestamp_s is None else observation_timestamp_s
            )
            request_id = uuid.uuid4().hex
            observation_id = f"{self.identity.session_id}-obs-{sequence_id}"
            metadata = {
                **self.identity.to_wire(),
                "sequence_id": sequence_id,
                "observation_id": observation_id,
                "observation_timestamp_s": timestamp_s,
            }
            raw = RawRequest(
                observation=dict(observation),
                prompt=prompt,
                metadata=metadata,
            )
            edge_request = InferenceRequest(
                payload=raw,
                request_id=request_id,
                seed=_session_sequence_seed(self.identity.session_id, sequence_id),
                metadata=metadata,
            )

            def build_cloud_request() -> InferenceRequest:
                # Snapshot storage without performing expensive CPU/list
                # serialization. A capture pipeline may instead transfer an
                # already immutable/reference-counted frame in O(1).
                cloud_observation = (
                    owned_cloud_observation
                    if owned_cloud_observation is not None
                    else _snapshot_observation(observation)
                )
                return replace(
                    edge_request,
                    payload=replace(
                        raw,
                        observation=cloud_observation,
                        metadata=dict(metadata),
                    ),
                    deadline_s=self.failover.config.cloud_request_timeout_s,
                )

            decision = await self.failover.infer_async(
                edge_request,
                cloud_request_factory=build_cloud_request,
            )
            if decision.source is ResultSource.EDGE:
                decision = replace(
                    decision,
                    result=replace(
                        decision.result,
                        metadata={
                            **decision.result.metadata,
                            **metadata,
                        },
                    ),
                )
            else:
                self._validate_cloud_result(
                    decision.result,
                    sequence_id=sequence_id,
                )
            return decision

    async def wait_for_cloud_idle(self) -> None:
        await self.failover.wait_for_cloud_idle()

    async def aclose(self) -> None:
        async with self._close_lock:
            if self._resources_closed:
                return
            self._closed = True
            # Let the admitted control tick finish before closing its Providers.
            async with self._inference_lock:
                pass
            reconnect = self._reconnect_task
            if reconnect is not None and not reconnect.done():
                reconnect.cancel()
                await asyncio.gather(reconnect, return_exceptions=True)
            await self.failover.aclose()
            await self.cloud.aclose(
                unregister_timeout_s=self.registration_timeout_s,
            )
            await self.edge.aclose()
            self._resources_closed = True

    def _schedule_reconnect(self) -> None:
        if self._closed or self.cloud.connected or not self.cloud.enabled:
            return
        active = self._reconnect_task
        if active is not None and not active.done():
            return
        now = asyncio.get_running_loop().time()
        if (
            self._last_reconnect_attempt_s is not None
            and now - self._last_reconnect_attempt_s < self.reconnect_interval_s
        ):
            return
        self._last_reconnect_attempt_s = now
        task = asyncio.create_task(
            self._connect_with_timeout(),
            name=f"cloud-register-{self.identity.session_id}",
        )
        self._reconnect_task = task

        def consume_result(completed: asyncio.Task[Mapping[str, Any]]) -> None:
            if self._reconnect_task is completed:
                self._reconnect_task = None
            try:
                completed.result()
            except asyncio.CancelledError:
                pass
            except Exception as error:  # noqa: BLE001 - retain background failure for diagnostics
                self._last_reconnect_error = error
            else:
                self._last_reconnect_error = None

        task.add_done_callback(consume_result)

    async def _connect_with_timeout(self) -> Mapping[str, Any]:
        return await asyncio.wait_for(
            self._connect_and_validate(),
            timeout=self.registration_timeout_s,
        )

    async def _connect_and_validate(self) -> Mapping[str, Any]:
        try:
            response = await self.cloud.connect()
            self._validate_cloud_registration(response)
        except CloudSessionRemoteError as error:
            if error.error_type == "SessionRegistrationError":
                self.cloud.set_enabled(False)
            raise
        except (CloudSessionProtocolError, ValueError):
            self.cloud.set_enabled(False)
            raise
        return response

    def _validate_cloud_registration(self, response: Mapping[str, Any]) -> None:
        capabilities = self.edge.capabilities
        expected = {
            "action_space_id": self.identity.action_space_id,
            "action_dim": capabilities.model.action_dim,
            "action_horizon": capabilities.model.action_horizon,
        }
        for name, value in expected.items():
            if response.get(name) != value:
                self.cloud.set_enabled(False)
                raise ValueError(
                    f"cloud {name} {response.get(name)!r} does not match edge value {value!r}"
                )
        supported = response.get("supported_embodiments")
        if not isinstance(supported, list) or self.identity.embodiment not in supported:
            self.cloud.set_enabled(False)
            raise ValueError(f"cloud does not support embodiment {self.identity.embodiment!r}")
        server_sequence = int(response.get("last_sequence_id", 0))
        if server_sequence > self._last_sequence_id:
            self.cloud.set_enabled(False)
            raise ValueError(
                f"cloud session is already at sequence {server_sequence}; "
                "rotate session_id instead of reusing an older session"
            )

    def _validate_cloud_result(
        self,
        result: InferenceResult,
        *,
        sequence_id: int,
    ) -> None:
        for name, expected in self.identity.to_wire().items():
            if result.metadata.get(name) != expected:
                raise RuntimeError(f"cloud result has mismatched {name}")
        source_sequence = int(result.metadata.get("sequence_id", 0))
        if not 0 < source_sequence <= sequence_id:
            raise RuntimeError(
                f"cloud result sequence {source_sequence} is invalid for "
                f"edge sequence {sequence_id}"
            )


def run_multi_robot_edge(config: MultiRobotEdgeConfig) -> list[dict[str, Any]]:
    if config.identity.session_id == "auto":
        config = replace(
            config,
            identity=RobotSessionIdentity(
                robot_id=config.identity.robot_id,
                edge_node_id=config.identity.edge_node_id,
                session_id=uuid.uuid4().hex,
                embodiment=config.identity.embodiment,
                action_space_id=config.identity.action_space_id,
            ),
        )
    edge = build_local_provider(config.provider)
    capabilities = edge.capabilities
    device = capabilities.device
    registration_metadata = {
        "physical_host_id": config.physical_host_id,
        "physical_resource_id": config.physical_resource_id,
        "edge_provider": capabilities.name,
        "edge_provider_runtime": capabilities.runtime,
        "edge_model_id": capabilities.model.model_id,
        "edge_model_family": capabilities.model.family,
        "edge_action_dim": capabilities.model.action_dim,
        "edge_action_horizon": capabilities.model.action_horizon,
        "edge_backend": device.backend if device is not None else None,
        "edge_device": device.device_id if device is not None else None,
    }
    cloud = MultiTenantTcpEndpoint(
        TcpJsonRequestClient(
            config.cloud_host,
            config.cloud_port,
            timeout_s=config.cloud_timeout_s,
        ),
        config.identity,
        registration_metadata=registration_metadata,
    )
    runtime = RobotEdgeRuntime(
        identity=config.identity,
        edge=edge,
        cloud=cloud,
        failover=config.failover,
        registration_timeout_s=config.registration_timeout_s,
        reconnect_interval_s=config.reconnect_interval_s,
    )
    return asyncio.run(_run_edge_demo(runtime, config))


async def _run_edge_demo(
    runtime: RobotEdgeRuntime,
    config: MultiRobotEdgeConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        await runtime.start()
        for tick in range(1, config.ticks + 1):
            if tick == config.disconnect_tick:
                runtime.set_cloud_enabled(False)
            if tick == config.reconnect_tick:
                runtime.set_cloud_enabled(True)
            observation = _build_demo_observation(runtime, config, tick=tick)
            decision = await runtime.infer_observation(
                observation,
                sequence_id=tick,
            )
            records.append(
                {
                    "robot_id": config.identity.robot_id,
                    "edge_node_id": config.identity.edge_node_id,
                    "session_id": config.identity.session_id,
                    "tick": tick,
                    "cloud_connected": runtime.cloud.connected,
                    "connection_epoch": decision.connection_epoch,
                    "source": decision.source.value,
                    "source_sequence_id": decision.source_sequence_id,
                    "cloud_result_age_s": decision.cloud_result_age_s,
                    "fallback_reason": (
                        decision.fallback_reason.value
                        if decision.fallback_reason is not None
                        else None
                    ),
                    "selected_observation_id": decision.result.metadata.get("observation_id"),
                    "selected_queue_time_s": decision.result.queue_time_s,
                    "selected_execution_time_s": decision.result.execution_time_s,
                    "output": json_compatible(
                        decision.result.output,
                        path="decision.output",
                    ),
                }
            )
            if config.control_period_s:
                await asyncio.sleep(config.control_period_s)
            else:
                await asyncio.sleep(0)
        if runtime.failover.cloud_request_in_flight:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    runtime.wait_for_cloud_idle(),
                    timeout=config.failover.cloud_request_timeout_s,
                )
        return records
    finally:
        await runtime.aclose()


def _build_demo_observation(
    runtime: RobotEdgeRuntime,
    config: MultiRobotEdgeConfig,
    *,
    tick: int,
) -> Mapping[str, Any]:
    if config.observation_mode == "target_vector":
        action_dim = runtime.edge.capabilities.model.action_dim
        if action_dim is None or action_dim <= 0:
            raise ValueError("target-vector demo requires an edge model action_dim")
        return {
            "target": [
                config.observation_offset + float(tick) + index / action_dim
                for index in range(action_dim)
            ]
        }

    adapter = getattr(runtime.edge, "adapter", None)
    synthetic_batch = getattr(adapter, "synthetic_batch", None)
    if not callable(synthetic_batch):
        raise TypeError(
            "adapter_synthetic observations require runtime.edge.adapter.synthetic_batch"
        )
    observation = synthetic_batch(
        batch_size=1,
        language_length=config.observation_language_length,
        seed=config.observation_seed + tick - 1,
    )
    if not isinstance(observation, Mapping):
        raise TypeError("adapter synthetic_batch must return an observation mapping")
    return observation


def _snapshot_observation(
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Take ownership of mutable frame storage without serializing it."""

    return {key: _snapshot_observation_value(value) for key, value in observation.items()}


def _snapshot_observation_value(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (str, bytes, bool, int, float),
    ):
        return value
    if isinstance(value, Mapping):
        return {key: _snapshot_observation_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_observation_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_observation_value(item) for item in value)

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    copy_value = getattr(candidate, "copy", None)
    if callable(copy_value):
        return copy_value()
    return copy.deepcopy(candidate)


def _session_sequence_seed(session_id: str, sequence_id: int) -> int:
    """Return a stable seed without relying on Python's randomized hash."""

    digest = hashlib.blake2b(
        f"{session_id}:{sequence_id}".encode(),
        digest_size=8,
        person=b"edge-cloud-seed",
    ).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one robot-scoped edge model against a shared cloud model."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_robot_edge.toml",
    )
    parser.add_argument("--robot-id")
    parser.add_argument("--edge-node-id")
    parser.add_argument("--session-id")
    parser.add_argument("--physical-host-id")
    parser.add_argument("--physical-resource-id")
    parser.add_argument("--cloud-host")
    parser.add_argument("--cloud-port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-base-path")
    parser.add_argument(
        "--observation-mode",
        choices=tuple(sorted(_OBSERVATION_MODES)),
    )
    parser.add_argument("--observation-offset", type=float)
    parser.add_argument("--observation-seed", type=int)
    parser.add_argument("--observation-language-length", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--disconnect-tick", type=int)
    parser.add_argument("--reconnect-tick", type=int)
    parser.add_argument(
        "--omit-output",
        action="store_true",
        help="omit the action matrix from CLI JSON while retaining source and timing records",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_multi_robot_edge_config(args.config)
    identity = RobotSessionIdentity(
        robot_id=args.robot_id or config.identity.robot_id,
        edge_node_id=args.edge_node_id or config.identity.edge_node_id,
        session_id=args.session_id or config.identity.session_id,
        embodiment=config.identity.embodiment,
        action_space_id=config.identity.action_space_id,
    )
    provider = config.provider
    if args.device is not None:
        provider = replace(provider, device=args.device)
    if args.checkpoint is not None:
        provider = replace(provider, checkpoint=args.checkpoint)
    if args.vlm_base_path is not None:
        package_options = dict(provider.package_options)
        package_options["vlm_base_path"] = args.vlm_base_path
        provider = replace(provider, package_options=package_options)
    config = replace(
        config,
        identity=identity,
        provider=provider,
        physical_host_id=args.physical_host_id or config.physical_host_id,
        physical_resource_id=(args.physical_resource_id or config.physical_resource_id),
        cloud_host=args.cloud_host or config.cloud_host,
        cloud_port=config.cloud_port if args.cloud_port is None else args.cloud_port,
        observation_mode=(
            config.observation_mode if args.observation_mode is None else args.observation_mode
        ),
        observation_offset=(
            config.observation_offset
            if args.observation_offset is None
            else args.observation_offset
        ),
        observation_seed=(
            config.observation_seed if args.observation_seed is None else args.observation_seed
        ),
        observation_language_length=(
            config.observation_language_length
            if args.observation_language_length is None
            else args.observation_language_length
        ),
        ticks=config.ticks if args.ticks is None else args.ticks,
        disconnect_tick=(
            config.disconnect_tick if args.disconnect_tick is None else args.disconnect_tick
        ),
        reconnect_tick=(
            config.reconnect_tick if args.reconnect_tick is None else args.reconnect_tick
        ),
    )
    records = run_multi_robot_edge(config)
    if args.omit_output:
        records = [
            {name: value for name, value in record.items() if name != "output"}
            for record in records
        ]
    print(json.dumps(records, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
