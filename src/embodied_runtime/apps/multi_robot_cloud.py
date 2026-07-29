"""Serve one loaded model to multiple isolated robot sessions."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import socket
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from embodied_runtime.contracts import InferenceRequest, RawRequest
from embodied_runtime.distributed.communication import (
    MULTI_ROBOT_PROTOCOL_VERSION,
    TcpJsonRequestClient,
    json_compatible,
    read_json_message,
    write_json_message,
)
from embodied_runtime.distributed.session import RobotSessionIdentity
from embodied_runtime.integrations.serving.multitenant import (
    MultiTenantInferenceService,
)

from ._local_provider import (
    LocalProviderConfig,
    build_local_provider,
)


@dataclass(frozen=True, slots=True)
class MultiRobotCloudConfig:
    provider: LocalProviderConfig = field(
        default_factory=lambda: LocalProviderConfig(
            provider_name="cloud-local",
            multi_tenant_safe=True,
            max_batch_size=1,
        )
    )
    host: str = "127.0.0.1"
    port: int = 18770
    request_timeout_s: float = 30.0
    max_sessions: int = 64
    max_pending_per_session: int = 1
    session_idle_ttl_s: float = 300.0
    action_space_id: str = "toy-vector-actions-v1"
    supported_embodiments: tuple[str, ...] = ("toy-vector",)
    shutdown_grace_s: float = 5.0

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("cloud host must not be empty")
        if not 0 < self.port < 65536:
            raise ValueError("cloud port must be between 1 and 65535")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and greater than zero")
        if self.max_sessions <= 0:
            raise ValueError("max_sessions must be greater than zero")
        if self.max_pending_per_session <= 0:
            raise ValueError("max_pending_per_session must be greater than zero")
        if not math.isfinite(self.session_idle_ttl_s) or self.session_idle_ttl_s <= 0:
            raise ValueError("session_idle_ttl_s must be finite and greater than zero")
        if not self.action_space_id.strip():
            raise ValueError("action_space_id must not be empty")
        if not self.supported_embodiments:
            raise ValueError("supported_embodiments must not be empty")
        if not math.isfinite(self.shutdown_grace_s) or self.shutdown_grace_s < 0:
            raise ValueError("shutdown_grace_s must be finite and non-negative")


def load_multi_robot_cloud_config(path: str | Path) -> MultiRobotCloudConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    service = raw.get("service", {})
    provider = raw.get("provider", {})
    if not isinstance(service, Mapping) or not isinstance(provider, Mapping):
        raise TypeError("service and provider must be TOML tables")
    supported_embodiments = service.get(
        "supported_embodiments",
        ("toy-vector",),
    )
    if not isinstance(supported_embodiments, (list, tuple)):
        raise TypeError("service.supported_embodiments must be an array")
    return MultiRobotCloudConfig(
        provider=LocalProviderConfig.from_mapping(provider),
        host=str(service.get("host", "127.0.0.1")),
        port=int(service.get("port", 18770)),
        request_timeout_s=float(service.get("request_timeout_s", 30.0)),
        max_sessions=int(service.get("max_sessions", 64)),
        max_pending_per_session=int(service.get("max_pending_per_session", 1)),
        session_idle_ttl_s=float(service.get("session_idle_ttl_s", 300.0)),
        action_space_id=str(service.get("action_space_id", "toy-vector-actions-v1")),
        supported_embodiments=tuple(str(value) for value in supported_embodiments),
        shutdown_grace_s=float(service.get("shutdown_grace_s", 5.0)),
    )


def run_multi_robot_cloud(config: MultiRobotCloudConfig) -> None:
    provider = build_local_provider(config.provider)
    service = MultiTenantInferenceService(
        provider,
        action_space_id=config.action_space_id,
        supported_embodiments=config.supported_embodiments,
        max_sessions=config.max_sessions,
        max_pending_per_session=config.max_pending_per_session,
        session_idle_ttl_s=config.session_idle_ttl_s,
    )
    try:
        asyncio.run(serve_multi_robot_cloud(service, config))
    except KeyboardInterrupt:
        pass


async def serve_multi_robot_cloud(
    service: MultiTenantInferenceService,
    config: MultiRobotCloudConfig,
) -> None:
    """Serve until cancelled and drain active handlers before closing Provider."""

    handlers: set[asyncio.Task[Any]] = set()

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            try:
                request = await read_json_message(
                    reader,
                    timeout_s=config.request_timeout_s,
                )
                response = await asyncio.wait_for(
                    dispatch_multi_robot_request(service, request),
                    timeout=config.request_timeout_s,
                )
            except Exception as error:  # noqa: BLE001 - serialize request failures to the peer
                response = {
                    "ok": False,
                    "kind": "error",
                    "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            try:
                await asyncio.wait_for(
                    write_json_message(writer, response),
                    timeout=config.request_timeout_s,
                )
            except (ConnectionError, OSError, asyncio.TimeoutError):
                pass
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    def accept(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # Register synchronously so shutdown cannot miss a handler that has
        # been accepted but has not yet started executing.
        task = asyncio.create_task(handle(reader, writer))
        handlers.add(task)

        def completed(done: asyncio.Task[Any]) -> None:
            handlers.discard(done)
            writer.close()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                done.result()

        task.add_done_callback(completed)

    server: asyncio.Server | None = None
    try:
        server = await asyncio.start_server(accept, config.host, config.port)
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
        capabilities = service.capabilities
        device = capabilities.device.device_id if capabilities.device is not None else "external"
        print(
            "MULTI_ROBOT_CLOUD_READY "
            f"host={socket.gethostname()} listen={addresses} "
            f"model={capabilities.model.model_id} device={device}",
            flush=True,
        )
        async with server:
            await server.serve_forever()
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
        await _drain_handlers(handlers, timeout_s=config.shutdown_grace_s)
        await service.aclose()


async def _drain_handlers(
    handlers: set[asyncio.Task[Any]],
    *,
    timeout_s: float,
) -> None:
    active = tuple(task for task in handlers if not task.done())
    if not active:
        return
    if timeout_s:
        done, pending = await asyncio.wait(active, timeout=timeout_s)
        del done
    else:
        pending = set(active)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


async def dispatch_multi_robot_request(
    service: MultiTenantInferenceService,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if request.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
        raise ValueError(f"protocol_version must be {MULTI_ROBOT_PROTOCOL_VERSION}")
    kind = request.get("kind")
    if kind == "ping":
        return {
            "ok": True,
            "kind": "pong",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **_provider_description(service),
            "registered_sessions": service.registered_session_count,
        }

    identity = RobotSessionIdentity.from_wire(request)
    if kind == "register":
        metadata = request.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("registration metadata must be an object")
        snapshot = service.register(identity, metadata=metadata)
        return {
            "ok": True,
            "kind": "registered",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **identity.to_wire(),
            **_provider_description(service),
            "last_sequence_id": snapshot.last_sequence_id,
            "registered_sessions": service.registered_session_count,
        }
    if kind == "unregister":
        service.unregister(identity)
        return {
            "ok": True,
            "kind": "unregistered",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
            **identity.to_wire(),
            "registered_sessions": service.registered_session_count,
        }
    if kind != "infer":
        raise ValueError(f"unsupported request kind: {kind!r}")

    request_id = str(request.get("request_id") or "")
    if not request_id:
        raise ValueError("infer request requires request_id")
    sequence_id = int(request.get("sequence_id", 0))
    observation_id = str(request.get("observation_id") or "")
    observation_timestamp_s = float(request.get("observation_timestamp_s", -1.0))
    observation = request.get("observation")
    if not isinstance(observation, Mapping):
        raise TypeError("infer request observation must be an object")
    request_metadata = request.get("request_metadata", {})
    raw_metadata = request.get("raw_metadata", {})
    if not isinstance(request_metadata, Mapping):
        raise TypeError("request_metadata must be an object")
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("raw_metadata must be an object")
    prompt = request.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("prompt must be a string or null")

    deadline = request.get("deadline_s")
    deadline_s = None if deadline is None else float(deadline)
    if deadline_s is not None and (not math.isfinite(deadline_s) or deadline_s <= 0):
        raise ValueError("deadline_s must be finite and greater than zero")
    num_steps_value = request.get("num_steps")
    if num_steps_value is not None:
        raise ValueError(
            "remote num_steps overrides are not admitted; configure the cloud Provider instead"
        )
    num_steps = None
    seed_value = request.get("seed")
    seed = None if seed_value is None else int(seed_value)

    result = await service.infer_async(
        identity,
        InferenceRequest(
            payload=RawRequest(
                observation=dict(observation),
                prompt=prompt,
                metadata=dict(raw_metadata),
            ),
            request_id=request_id,
            deadline_s=deadline_s,
            num_steps=num_steps,
            seed=seed,
            metadata=dict(request_metadata),
        ),
        sequence_id=sequence_id,
        observation_id=observation_id,
        observation_timestamp_s=observation_timestamp_s,
    )
    output, metadata = await asyncio.gather(
        asyncio.to_thread(
            json_compatible,
            result.output,
            path="result.output",
        ),
        asyncio.to_thread(
            json_compatible,
            result.metadata,
            path="result.metadata",
        ),
    )
    return {
        "ok": True,
        "kind": "inference_result",
        "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
        **identity.to_wire(),
        "request_id": result.request_id,
        "sequence_id": sequence_id,
        "observation_id": observation_id,
        "status": result.status.value,
        "output": output,
        "queue_time_s": result.queue_time_s,
        "execution_time_s": result.execution_time_s,
        "metadata": metadata,
    }


def _provider_description(
    service: MultiTenantInferenceService,
) -> dict[str, Any]:
    capabilities = service.capabilities
    device = capabilities.device
    return {
        "provider": capabilities.name,
        "provider_runtime": capabilities.runtime,
        "model_id": capabilities.model.model_id,
        "model_family": capabilities.model.family,
        "model_revision": capabilities.model.revision,
        "action_space_id": service.contract.action_space_id,
        "supported_embodiments": sorted(service.contract.supported_embodiments),
        "action_dim": service.contract.action_dim,
        "action_horizon": service.contract.action_horizon,
        "backend": device.backend if device is not None else None,
        "device": device.device_id if device is not None else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one model to multiple isolated robot sessions."
    )
    parser.add_argument(
        "--config",
        default="configs/multi_robot_cloud.toml",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--checkpoint")
    parser.add_argument("--vlm-base-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_multi_robot_cloud_config(args.config)
    provider = config.provider
    if args.device is not None:
        provider = replace(provider, device=args.device)
    if args.checkpoint is not None:
        provider = replace(provider, checkpoint=args.checkpoint)
    if args.vlm_base_path is not None:
        provider = replace(
            provider,
            package_options={
                **provider.package_options,
                "vlm_base_path": args.vlm_base_path,
            },
        )
    config = replace(
        config,
        provider=provider,
        host=config.host if args.host is None else args.host,
        port=config.port if args.port is None else args.port,
    )
    run_multi_robot_cloud(config)
    return 0


def _required_non_empty_string(
    response: Mapping[str, Any],
    key: str,
) -> str:
    value = response.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"health response requires non-empty {key}")
    return value


def _optional_string(
    response: Mapping[str, Any],
    key: str,
) -> str | None:
    value = response.get(key)
    if value is not None and not isinstance(value, str):
        raise TypeError(f"health response {key} must be a string or null")
    return value


def _optional_positive_integer(
    response: Mapping[str, Any],
    key: str,
) -> int | None:
    value = response.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"health response {key} must be a positive integer or null")
    return value


def validate_multi_robot_health_response(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize the read-only cloud ``ping`` response."""

    if response.get("ok") is not True:
        error_type = str(response.get("error_type") or "CloudHealthError")
        error = str(response.get("error") or "cloud reported an unhealthy status")
        raise RuntimeError(f"{error_type}: {error}")
    if response.get("kind") != "pong":
        raise ValueError("health response kind must be 'pong'")
    if response.get("protocol_version") != MULTI_ROBOT_PROTOCOL_VERSION:
        raise ValueError(f"health response protocol_version must be {MULTI_ROBOT_PROTOCOL_VERSION}")

    embodiments = response.get("supported_embodiments")
    if (
        not isinstance(embodiments, list)
        or not embodiments
        or any(not isinstance(value, str) or not value.strip() for value in embodiments)
    ):
        raise ValueError("health response supported_embodiments must be a non-empty string array")
    registered_sessions = response.get("registered_sessions")
    if (
        isinstance(registered_sessions, bool)
        or not isinstance(registered_sessions, int)
        or registered_sessions < 0
    ):
        raise ValueError("health response registered_sessions must be a non-negative integer")

    return {
        "healthy": True,
        "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
        "provider": {
            "name": _required_non_empty_string(response, "provider"),
            "runtime": _required_non_empty_string(response, "provider_runtime"),
            "backend": _optional_string(response, "backend"),
            "device": _optional_string(response, "device"),
        },
        "model": {
            "id": _required_non_empty_string(response, "model_id"),
            "family": _required_non_empty_string(response, "model_family"),
            "revision": _optional_string(response, "model_revision"),
        },
        "action_contract": {
            "action_space_id": _required_non_empty_string(response, "action_space_id"),
            "supported_embodiments": sorted(embodiments),
            "action_dim": _optional_positive_integer(response, "action_dim"),
            "action_horizon": _optional_positive_integer(response, "action_horizon"),
        },
        "registered_sessions": registered_sessions,
    }


async def probe_multi_robot_cloud_health(
    host: str,
    port: int,
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Probe one cloud endpoint without registering a robot or running inference."""

    response = await TcpJsonRequestClient(
        host,
        port,
        timeout_s=timeout_s,
    ).request(
        {
            "kind": "ping",
            "protocol_version": MULTI_ROBOT_PROTOCOL_VERSION,
        }
    )
    return validate_multi_robot_health_response(response)


def build_health_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only health probe for a multi-robot cloud service."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18770)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    return parser


def health_main(argv: Sequence[str] | None = None) -> int:
    args = build_health_parser().parse_args(argv)
    try:
        health = asyncio.run(
            probe_multi_robot_cloud_health(
                args.host,
                args.port,
                timeout_s=args.timeout_s,
            )
        )
    except Exception as error:  # noqa: BLE001 - any failed probe is an unhealthy CLI result
        print(
            json.dumps(
                {
                    "healthy": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(health, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
