"""Compose two local engines behind an asynchronous cloud/edge failover policy."""

from __future__ import annotations

import argparse
import asyncio
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import CompileOptions, InferenceRequest, RawRequest
from embodied_runtime.distributed import (
    AsyncFailoverCoordinator,
    FailoverConfig,
    FailoverMode,
)
from embodied_runtime.distributed.communication import (
    DummyLink,
    DummyRemoteInferenceEndpoint,
)
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.toy_single_forward import ToySingleForwardAdapter


@dataclass(frozen=True, slots=True)
class DemoConfig:
    ticks: int = 12
    control_period_s: float = 0.05
    disconnect_tick: int | None = 5
    reconnect_tick: int | None = 8
    one_way_latency_s: float = 0.04
    failover: FailoverConfig = FailoverConfig(max_cloud_sequence_lag=4)

    def __post_init__(self) -> None:
        if self.ticks <= 0:
            raise ValueError("ticks must be greater than zero")
        if self.control_period_s < 0:
            raise ValueError("control_period_s cannot be negative")
        if self.one_way_latency_s < 0:
            raise ValueError("one_way_latency_s cannot be negative")


def load_demo_config(path: str | Path) -> DemoConfig:
    """Load the example's policy and fault schedule from TOML."""

    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    failover_values = raw.get("failover", {})
    demo_values = raw.get("demo", {})
    link_values = raw.get("dummy_link", {})
    return DemoConfig(
        ticks=int(demo_values.get("ticks", 12)),
        control_period_s=float(demo_values.get("control_period_s", 0.05)),
        disconnect_tick=_optional_int(demo_values.get("disconnect_tick", 5)),
        reconnect_tick=_optional_int(demo_values.get("reconnect_tick", 8)),
        one_way_latency_s=float(link_values.get("one_way_latency_s", 0.04)),
        failover=FailoverConfig(
            mode=FailoverMode(failover_values.get("mode", FailoverMode.ASYNC_CLOUD_PREFERRED)),
            cloud_request_timeout_s=float(failover_values.get("cloud_request_timeout_s", 5.0)),
            cloud_result_ttl_s=float(failover_values.get("cloud_result_ttl_s", 1.0)),
            max_cloud_sequence_lag=int(failover_values.get("max_cloud_sequence_lag", 4)),
            cloud_submit_interval_s=float(failover_values.get("cloud_submit_interval_s", 0.0)),
        ),
    )


async def run_cloud_edge_failover_demo(config: DemoConfig) -> list[dict[str, Any]]:
    """Exercise result-level failover with two real ExecutionEngine instances."""

    edge_adapter, edge_engine = _build_toy_engine()
    cloud_adapter, cloud_engine = _build_toy_engine()
    link = DummyLink(one_way_latency_s=config.one_way_latency_s)
    remote_cloud = DummyRemoteInferenceEndpoint(cloud_engine, link)
    coordinator = AsyncFailoverCoordinator(
        edge=edge_engine,
        cloud=remote_cloud,
        config=config.failover,
    )
    records: list[dict[str, Any]] = []

    await edge_engine.start()
    await cloud_engine.start()
    try:
        for tick in range(1, config.ticks + 1):
            if tick == config.disconnect_tick:
                link.set_connected(False)
            if tick == config.reconnect_tick:
                link.set_connected(True)

            raw_request = RawRequest(observation={"target": [float(tick), -float(tick)]})
            edge_request = InferenceRequest(
                payload=edge_adapter.preprocess_one(raw_request),
                request_id=f"edge-{tick}",
            )
            cloud_request = InferenceRequest(
                payload=cloud_adapter.preprocess_one(raw_request),
                request_id=f"cloud-{tick}",
            )
            decision = await coordinator.infer_async(
                edge_request,
                cloud_request=cloud_request,
            )
            selected_adapter = cloud_adapter if decision.source.value == "cloud" else edge_adapter
            action = selected_adapter.postprocess_one(decision.result.output)
            records.append(
                {
                    "tick": tick,
                    "connected": link.connected,
                    "connection_epoch": decision.connection_epoch,
                    "source": decision.source.value,
                    "source_sequence_id": decision.source_sequence_id,
                    "fallback_reason": (
                        decision.fallback_reason.value
                        if decision.fallback_reason is not None
                        else None
                    ),
                    "actions": action.actions,
                }
            )
            if config.control_period_s:
                await asyncio.sleep(config.control_period_s)
            else:
                await asyncio.sleep(0)
    finally:
        await coordinator.aclose()
        await asyncio.gather(edge_engine.aclose(), cloud_engine.aclose())

    return records


def _build_toy_engine() -> tuple[ToySingleForwardAdapter, ExecutionEngine]:
    adapter = ToySingleForwardAdapter(action_horizon=1, action_dim=2)
    package = adapter.build_package()
    backend = TorchCudaBackend()
    cpu = next(device for device in backend.probe() if device.device_id == "cpu")
    artifact = backend.compile(
        package,
        cpu,
        CompileOptions(mode="eager", dtype="float32"),
    )
    engine = ExecutionEngine(
        package,
        backend.load(artifact),
        batcher=adapter.collate,
        splitter=adapter.unbatch,
    )
    return adapter, engine


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _format_actions(actions: Any) -> str:
    if hasattr(actions, "tolist"):
        actions = actions.tolist()
    return str(actions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configurable asynchronous cloud/edge failover demo."
    )
    parser.add_argument(
        "--config",
        default="configs/cloud_edge_failover.toml",
        help="TOML policy and fault schedule",
    )
    parser.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in FailoverMode),
        help="override failover.mode from the TOML file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_demo_config(args.config)
    if args.mode is not None:
        config = replace(
            config,
            failover=replace(config.failover, mode=FailoverMode(args.mode)),
        )
    records = asyncio.run(run_cloud_edge_failover_demo(config))

    print("tick  link  source  source_seq  reason                 actions")
    for record in records:
        link = "up" if record["connected"] else "down"
        reason = record["fallback_reason"] or "-"
        print(
            f"{record['tick']:>4}  {link:<4}  {record['source']:<6}  "
            f"{record['source_sequence_id']:>10}  {reason:<21}  "
            f"{_format_actions(record['actions'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
