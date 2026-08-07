"""Async TCP server lifecycle for multi-robot inference."""

from __future__ import annotations

import asyncio
import contextlib
import socket
from typing import Any

from embodied_runtime.distributed.communication import (
    MULTI_ROBOT_PROTOCOL_VERSION,
    read_json_message,
    write_json_message,
)
from embodied_runtime.distributed.multitenant import MultiTenantInferenceService

from .._local_provider import build_local_provider
from .cloud_codec import dispatch_multi_robot_request
from .cloud_settings import MultiRobotCloudConfig


def run_multi_robot_cloud(config: MultiRobotCloudConfig) -> None:
    """Build the configured provider and serve it until interrupted."""

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
                request = await read_json_message(reader, timeout_s=config.request_timeout_s)
                response = await asyncio.wait_for(
                    dispatch_multi_robot_request(service, request),
                    timeout=config.request_timeout_s,
                )
            except Exception as error:  # noqa: BLE001 - serialize failures to peer
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

    def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Register synchronously so shutdown cannot miss an accepted handler.
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
        await drain_handlers(handlers, timeout_s=config.shutdown_grace_s)
        await service.aclose()


async def drain_handlers(
    handlers: set[asyncio.Task[Any]],
    *,
    timeout_s: float,
) -> None:
    active = tuple(task for task in handlers if not task.done())
    if not active:
        return
    if timeout_s:
        _, pending = await asyncio.wait(active, timeout=timeout_s)
    else:
        pending = set(active)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
