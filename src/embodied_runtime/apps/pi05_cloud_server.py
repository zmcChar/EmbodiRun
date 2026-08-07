"""Serve a real π0.5 CUDA runtime over the prototype TCP/JSON transport."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embodied_runtime.distributed.communication import (
    read_json_message,
    write_json_message,
)
from embodied_runtime.engine.request import InferenceRequest

from .cloud_edge.pi05_settings import load_pi05_cpu_gpu_config


@dataclass(frozen=True, slots=True)
class Pi05CloudServerConfig:
    checkpoint: str
    device: str = "cuda:0"
    dtype: str = "preserve"
    num_steps: int = 10
    language_length: int = 8
    seed: int = 0
    cuda_graph: bool = False
    local_files_only: bool = True
    host: str = "0.0.0.0"
    port: int = 18765
    request_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint must not be empty")
        if not self.device.startswith("cuda:"):
            raise ValueError("π0.5 cloud server device must be cuda:N")
        if self.dtype not in {"preserve", "float32", "float16", "bfloat16"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        if self.num_steps <= 0 or self.language_length <= 0:
            raise ValueError("num_steps and language_length must be greater than zero")
        if not self.host:
            raise ValueError("host must not be empty")
        if not 0 < self.port < 65536:
            raise ValueError("port must be between 1 and 65535")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be greater than zero")


@dataclass(slots=True)
class _CloudRuntime:
    adapter: Any
    engine: Any
    payload: Mapping[str, Any]
    _closed: bool = False

    def close(self) -> None:
        if not self._closed:
            self.engine.close()
            self._closed = True


def run_pi05_cloud_server(config: Pi05CloudServerConfig) -> None:
    """Load the CUDA model once and serve until interrupted."""

    runtime = _build_cloud_runtime(config)
    try:
        asyncio.run(_serve(runtime, config))
    except KeyboardInterrupt:
        pass
    finally:
        runtime.close()


def _build_cloud_runtime(
    config: Pi05CloudServerConfig,
    *,
    adapter_factory: Callable[[], Any] | None = None,
    backend_factory: Callable[[], Any] | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> _CloudRuntime:
    from embodied_runtime.backends.compile import CompileOptions

    if adapter_factory is None:
        from embodied_runtime.models.vla.pi05 import Pi05Adapter

        adapter_factory = Pi05Adapter
    if backend_factory is None:
        from embodied_runtime.backends.torch_cuda import TorchCudaBackend

        backend_factory = TorchCudaBackend
    if engine_factory is None:
        from embodied_runtime.engine import ExecutionEngine

        engine_factory = ExecutionEngine

    adapter = adapter_factory()
    package = adapter.build_package(
        config.checkpoint,
        local_files_only=config.local_files_only,
    )
    backend = backend_factory()
    devices = {device.device_id: device for device in backend.probe()}
    try:
        device = devices[config.device]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"device {config.device!r} is unavailable; detected: {available}"
        ) from error

    options: dict[str, Any] = {"empty_cache_on_close": True}
    if config.cuda_graph:
        options["cuda_graph_entrypoints"] = (package.plan.step,)
        options["cuda_graph_dynamic_scalar_inputs"] = {
            package.plan.step: ("time",),
        }
    artifact = backend.compile(
        package,
        device,
        CompileOptions(
            mode="eager",
            dtype=None if config.dtype == "preserve" else config.dtype,
            options=options,
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
    return _CloudRuntime(adapter=adapter, engine=engine, payload=payload)


async def _serve(
    runtime: _CloudRuntime,
    config: Pi05CloudServerConfig,
) -> None:
    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        try:
            request = await read_json_message(
                reader,
                timeout_s=config.request_timeout_s,
            )
            response = await _dispatch(runtime, config, request)
        except Exception as error:  # noqa: BLE001 - serialize request failure to peer
            response = {
                "ok": False,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        try:
            await write_json_message(writer, response)
        except (ConnectionError, OSError):
            return
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()
        if response.get("ok"):
            print(
                f"served request_id={response.get('request_id')} peer={peer} "
                f"execution_time_s={response.get('execution_time_s')}",
                flush=True,
            )

    server: asyncio.Server | None = None
    try:
        await runtime.engine.start()
        server = await asyncio.start_server(handle, config.host, config.port)
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
        print(
            "PI05_CLOUD_READY "
            f"host={socket.gethostname()} listen={addresses} "
            f"device={runtime.engine.session.device.device_id}",
            flush=True,
        )
        async with server:
            await server.serve_forever()
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
        await runtime.engine.aclose()
        runtime._closed = True


async def _dispatch(
    runtime: _CloudRuntime,
    config: Pi05CloudServerConfig,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    kind = request.get("kind")
    if kind == "ping":
        return {
            "ok": True,
            "kind": "pong",
            "hostname": socket.gethostname(),
            "model_id": runtime.engine.package.spec.model_id,
            "device": runtime.engine.session.device.device_id,
        }
    if kind != "infer":
        raise ValueError(f"unsupported request kind: {kind!r}")

    request_id = str(request.get("request_id") or "")
    if not request_id:
        raise ValueError("infer request requires request_id")
    num_steps = int(request.get("num_steps", config.num_steps))
    seed = int(request.get("seed", config.seed))
    if num_steps <= 0:
        raise ValueError("num_steps must be greater than zero")
    if num_steps > config.num_steps:
        raise ValueError(
            f"num_steps {num_steps} exceeds configured server limit {config.num_steps}"
        )

    result = await runtime.engine.infer_async(
        InferenceRequest(
            payload=runtime.payload,
            request_id=request_id,
            num_steps=num_steps,
            seed=seed,
        )
    )
    output = result.output
    actions = output["actions"] if isinstance(output, Mapping) else output
    actions_cpu = actions.detach().float().cpu()
    return {
        "ok": True,
        "kind": "inference_result",
        "request_id": result.request_id,
        "actions": actions_cpu.tolist(),
        "action_shape": list(actions_cpu.shape),
        "model_id": result.metadata.get("model_id"),
        "device": result.metadata.get("device_id"),
        "execution_time_s": result.execution_time_s,
        "hostname": socket.gethostname(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve π0.5 CUDA inference over TCP/JSON.")
    parser.add_argument(
        "--config",
        default="configs/pi05_cpu_gpu_collaboration.toml",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = load_pi05_cpu_gpu_config(args.config)
    config = Pi05CloudServerConfig(
        checkpoint=args.checkpoint or base.checkpoint,
        device=base.cloud_device,
        dtype=base.dtype,
        num_steps=base.num_steps if args.num_steps is None else args.num_steps,
        language_length=base.language_length,
        seed=base.seed,
        cuda_graph=args.cuda_graph or base.cuda_graph,
        local_files_only=False if args.allow_download else base.local_files_only,
        host=args.host,
        port=args.port,
        request_timeout_s=base.failover.cloud_request_timeout_s + 5.0,
    )
    run_pi05_cloud_server(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
