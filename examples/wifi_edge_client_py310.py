#!/usr/bin/env python3
"""Standalone Python 3.10 edge client for the real two-host Wi-Fi prototype.

The file intentionally does not import ``embodied_runtime``.  It only requires
the Python standard library and PyTorch, so it can be copied directly to the
RTX 3070 laptop.  Messages use the same four-byte big-endian length-prefixed
JSON protocol as ``embodied_runtime.distributed.communication.tcp_json``.
"""

import argparse
import asyncio
import json
import socket
import struct
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


_HEADER = struct.Struct("!I")
_MAX_MESSAGE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ClientConfig:
    cloud_host: str
    cloud_port: int = 18765
    edge_device: str = "cuda:0"
    cloud_weight: float = 0.5
    timeout_s: float = 120.0
    num_steps: int = 1
    seed: int = 0
    action_horizon: int = 50
    action_dim: int = 32
    observation_dim: int = 64
    hidden_dim: int = 128
    failure_probe_port: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.cloud_host:
            raise ValueError("cloud_host must not be empty")
        if not 0 < self.cloud_port < 65536:
            raise ValueError("cloud_port must be between 1 and 65535")
        if self.failure_probe_port is not None and not 0 < self.failure_probe_port < 65536:
            raise ValueError("failure_probe_port must be between 1 and 65535")
        if not 0.0 <= self.cloud_weight <= 1.0:
            raise ValueError("cloud_weight must be between zero and one")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if (
            min(
                self.action_horizon,
                self.action_dim,
                self.observation_dim,
                self.hidden_dim,
            )
            <= 0
        ):
            raise ValueError("model dimensions must be greater than zero")


class LightweightEdgePolicy(torch.nn.Module):
    """Small temporal MLP policy that runs as a real model on the edge GPU."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dim: int,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(observation_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
        )
        self.action_center = torch.nn.Linear(hidden_dim, action_dim)
        self.action_trend = torch.nn.Linear(hidden_dim, action_dim)
        self.register_buffer(
            "time_axis",
            torch.linspace(-1.0, 1.0, action_horizon).unsqueeze(1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(observation)
        center = self.action_center(latent)
        trend = self.action_trend(latent)
        return torch.tanh(center.unsqueeze(0) + self.time_axis * trend.unsqueeze(0))


async def _read_json_message(
    reader: asyncio.StreamReader,
    timeout_s: float,
) -> Tuple[Dict[str, Any], int]:
    header = await asyncio.wait_for(reader.readexactly(_HEADER.size), timeout=timeout_s)
    (size,) = _HEADER.unpack(header)
    if size > _MAX_MESSAGE_BYTES:
        raise RuntimeError(
            "cloud response size {} exceeds limit {}".format(size, _MAX_MESSAGE_BYTES)
        )
    encoded = await asyncio.wait_for(reader.readexactly(size), timeout=timeout_s)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("cloud returned invalid JSON: {}".format(error)) from error
    if not isinstance(value, dict):
        raise RuntimeError("cloud response must be a JSON object")
    return value, size


async def _write_json_message(
    writer: asyncio.StreamWriter,
    payload: Dict[str, Any],
    timeout_s: float,
) -> int:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise RuntimeError(
            "request size {} exceeds limit {}".format(len(encoded), _MAX_MESSAGE_BYTES)
        )
    writer.write(_HEADER.pack(len(encoded)))
    writer.write(encoded)
    await asyncio.wait_for(writer.drain(), timeout=timeout_s)
    return len(encoded)


async def _request_cloud(
    config: ClientConfig,
    request_id: str,
) -> Tuple[Dict[str, Any], float, int, int]:
    started = time.perf_counter()
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(config.cloud_host, config.cloud_port),
        timeout=config.timeout_s,
    )
    try:
        request_bytes = await _write_json_message(
            writer,
            {
                "kind": "infer",
                "request_id": request_id,
                "num_steps": config.num_steps,
                "seed": config.seed,
            },
            config.timeout_s,
        )
        response, response_bytes = await _read_json_message(reader, config.timeout_s)
    finally:
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()
    round_trip_time_s = time.perf_counter() - started
    if not response.get("ok"):
        error_type = response.get("error_type", "CloudError")
        error = response.get("error", "cloud request failed")
        raise RuntimeError("{}: {}".format(error_type, error))
    if response.get("kind") != "inference_result":
        raise RuntimeError("unexpected cloud response kind: {!r}".format(response.get("kind")))
    if response.get("request_id") != request_id:
        raise RuntimeError("cloud response request_id does not match the request")
    return response, round_trip_time_s, request_bytes, response_bytes


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_observation(config: ClientConfig, device: torch.device, tick: int) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, config.observation_dim, device=device)
    return torch.sin(axis * 2.0 + float(tick) * 0.1)


def _edge_tick(
    model: LightweightEdgePolicy,
    config: ClientConfig,
    device: torch.device,
    tick: int,
) -> Tuple[torch.Tensor, float]:
    observation = _make_observation(config, device, tick)
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        actions = model(observation)
    _synchronize(device)
    return actions, time.perf_counter() - started


def _cloud_actions_on_edge(
    response: Dict[str, Any],
    edge_actions: torch.Tensor,
) -> torch.Tensor:
    if "actions" not in response:
        raise RuntimeError("cloud response does not contain actions")
    cloud_actions = torch.tensor(
        response["actions"],
        dtype=edge_actions.dtype,
        device=edge_actions.device,
    )
    if cloud_actions.ndim == 3 and cloud_actions.shape[0] == 1:
        cloud_actions = cloud_actions.squeeze(0)
    if tuple(cloud_actions.shape) != tuple(edge_actions.shape):
        raise RuntimeError(
            "cloud action shape {} does not match edge action shape {}".format(
                tuple(cloud_actions.shape),
                tuple(edge_actions.shape),
            )
        )
    if not bool(torch.isfinite(cloud_actions).all().item()):
        raise RuntimeError("cloud actions contain non-finite values")
    return cloud_actions


async def run_wifi_edge_client(config: ClientConfig) -> Dict[str, Any]:
    """Run edge-first async inference and fall back to edge on any cloud error."""

    device = torch.device(config.edge_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA edge device requested, but torch.cuda.is_available() is false")

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = LightweightEdgePolicy(
        observation_dim=config.observation_dim,
        hidden_dim=config.hidden_dim,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
    ).to(device)
    model.eval()

    # Warm up the small edge policy before timing the control ticks.
    _edge_tick(model, config, device, tick=0)
    request_id = "wifi-{}".format(uuid.uuid4().hex)

    first_tick_started = time.perf_counter()
    cloud_request_started = first_tick_started
    cloud_task = asyncio.create_task(_request_cloud(config, request_id))
    first_edge, first_edge_execution_time_s = await asyncio.to_thread(
        _edge_tick,
        model,
        config,
        device,
        1,
    )
    first_tick_time_s = time.perf_counter() - first_tick_started
    cloud_in_flight_after_first = not cloud_task.done()

    response: Dict[str, Any] = {}
    cloud_rtt_s: Optional[float] = None
    request_bytes: Optional[int] = None
    response_bytes: Optional[int] = None
    cloud_error: Optional[Exception] = None
    try:
        response, cloud_rtt_s, request_bytes, response_bytes = await cloud_task
    except Exception as error:
        cloud_error = error
        cloud_rtt_s = time.perf_counter() - cloud_request_started

    second_edge, second_edge_execution_time_s = await asyncio.to_thread(
        _edge_tick,
        model,
        config,
        device,
        2,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    edge_gpu_name: Optional[str] = None
    if device.type == "cuda":
        edge_gpu_name = torch.cuda.get_device_name(device)

    summary: Dict[str, Any] = {
        "protocol": "length_prefixed_json_v1",
        "mode": "async_blend",
        "cloud_host": config.cloud_host,
        "cloud_port": config.cloud_port,
        "cloud_hostname": response.get("hostname"),
        "cloud_model_id": response.get("model_id"),
        "cloud_device": response.get("device"),
        "edge_hostname": socket.gethostname(),
        "edge_model_id": "lightweight-temporal-mlp",
        "edge_model_parameters": parameter_count,
        "edge_device": str(device),
        "edge_gpu_name": edge_gpu_name,
        "torch_version": torch.__version__,
        "request_id": request_id,
        "first_tick_source": "edge",
        "first_tick_time_s": first_tick_time_s,
        "first_edge_execution_time_s": first_edge_execution_time_s,
        "cloud_in_flight_after_first": cloud_in_flight_after_first,
        "cloud_round_trip_time_s": cloud_rtt_s,
        "cloud_server_execution_time_s": response.get("execution_time_s"),
        "cloud_request_succeeded": cloud_error is None,
        "second_tick_source": "edge",
        "second_edge_execution_time_s": second_edge_execution_time_s,
        "fallback_reason": "cloud_error" if cloud_error is not None else None,
        "cloud_error_type": type(cloud_error).__name__ if cloud_error is not None else None,
        "cloud_error": str(cloud_error) if cloud_error is not None else None,
        "fusion_time_s": None,
        "cloud_weight": config.cloud_weight,
        "action_shape": list(second_edge.shape),
        "final_action_device": str(second_edge.device),
        "fusion_max_abs_error": None,
        "fused_vs_edge_max_abs": None,
        "fused_vs_cloud_max_abs": None,
        "request_payload_bytes": request_bytes,
        "response_payload_bytes": response_bytes,
        "first_action_mean": first_edge.float().mean().item(),
        "final_action_mean": second_edge.float().mean().item(),
    }

    if cloud_error is None:
        try:
            cloud_actions = _cloud_actions_on_edge(response, second_edge)
            _synchronize(device)
            fusion_started = time.perf_counter()
            fused = second_edge + (cloud_actions - second_edge) * config.cloud_weight
            _synchronize(device)
            fusion_time_s = time.perf_counter() - fusion_started

            reference = (
                second_edge.double() * (1.0 - config.cloud_weight)
                + cloud_actions.double() * config.cloud_weight
            )
            summary.update(
                {
                    "second_tick_source": "blended",
                    "fusion_time_s": fusion_time_s,
                    "action_shape": list(fused.shape),
                    "final_action_device": str(fused.device),
                    "fusion_max_abs_error": (fused.double() - reference).abs().max().item(),
                    "fused_vs_edge_max_abs": (fused - second_edge).abs().max().item(),
                    "fused_vs_cloud_max_abs": (fused - cloud_actions).abs().max().item(),
                    "final_action_mean": fused.float().mean().item(),
                }
            )
        except Exception as error:
            summary.update(
                {
                    "cloud_request_succeeded": False,
                    "fallback_reason": "cloud_error",
                    "cloud_error_type": type(error).__name__,
                    "cloud_error": str(error),
                }
            )

    if config.failure_probe_port is not None:
        summary.update(
            await _run_failure_probe(
                config,
                model,
                device,
            )
        )
    return summary


async def _run_failure_probe(
    config: ClientConfig,
    model: LightweightEdgePolicy,
    device: torch.device,
) -> Dict[str, Any]:
    """Run a real bad-port request while still producing the next edge action."""

    assert config.failure_probe_port is not None
    probe_config = replace(
        config,
        cloud_port=config.failure_probe_port,
        failure_probe_port=None,
    )
    request_id = "failure-probe-{}".format(uuid.uuid4().hex)
    started = time.perf_counter()
    cloud_task = asyncio.create_task(_request_cloud(probe_config, request_id))
    edge_actions, edge_execution_time_s = await asyncio.to_thread(
        _edge_tick,
        model,
        config,
        device,
        3,
    )
    edge_tick_time_s = time.perf_counter() - started
    cloud_in_flight_after_edge = not cloud_task.done()
    cloud_error: Optional[Exception] = None
    try:
        await cloud_task
    except Exception as error:
        cloud_error = error

    return {
        "failure_probe_port": config.failure_probe_port,
        "failure_probe_tick_source": "edge",
        "failure_probe_reason": "cloud_error" if cloud_error is not None else None,
        "failure_probe_cloud_succeeded": cloud_error is None,
        "failure_probe_cloud_error_type": (
            type(cloud_error).__name__ if cloud_error is not None else None
        ),
        "failure_probe_cloud_error": (str(cloud_error) if cloud_error is not None else None),
        "failure_probe_cloud_in_flight_after_edge": cloud_in_flight_after_edge,
        "failure_probe_edge_tick_time_s": edge_tick_time_s,
        "failure_probe_edge_execution_time_s": edge_execution_time_s,
        "failure_probe_elapsed_time_s": time.perf_counter() - started,
        "failure_probe_action_shape": list(edge_actions.shape),
        "failure_probe_action_device": str(edge_actions.device),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a standalone RTX edge policy while π0.5 inference executes "
            "asynchronously on a remote cloud host."
        )
    )
    parser.add_argument("--cloud-host", required=True)
    parser.add_argument("--cloud-port", type=int, default=18765)
    parser.add_argument("--edge-device", default="cuda:0")
    parser.add_argument("--cloud-weight", type=float, default=0.5)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--observation-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--failure-probe-port",
        type=int,
        help="After a successful run, probe this expected-unreachable TCP port.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ClientConfig(
        cloud_host=args.cloud_host,
        cloud_port=args.cloud_port,
        edge_device=args.edge_device,
        cloud_weight=args.cloud_weight,
        timeout_s=args.timeout_s,
        num_steps=args.num_steps,
        seed=args.seed,
        action_horizon=args.action_horizon,
        action_dim=args.action_dim,
        observation_dim=args.observation_dim,
        hidden_dim=args.hidden_dim,
        failure_probe_port=args.failure_probe_port,
    )
    summary = asyncio.run(run_wifi_edge_client(config))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
