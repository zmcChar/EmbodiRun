"""Run the staged toy flow through the real engine/backend boundary."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import CompileOptions, RawRequest
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.toy_flow import ToyFlowAdapter


def _select_device(backend: TorchCudaBackend, device_id: str):
    devices = {device.device_id: device for device in backend.probe()}
    try:
        return devices[device_id]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"device {device_id!r} is unavailable; detected devices: {available}"
        ) from error


def run_toy_flow(
    *,
    target: Sequence[float] = (0.25, -0.5),
    device: str = "cpu",
    dtype: str | None = "float32",
    mode: str = "eager",
    num_steps: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Compose Group 1, Group 3, and Group 4 and return one inference summary."""

    adapter = ToyFlowAdapter(action_horizon=3, action_dim=len(target))
    package = adapter.build_package()
    payload = adapter.preprocess_one(RawRequest(observation={"target": list(target)}))

    backend = TorchCudaBackend()
    selected = _select_device(backend, device)
    artifact = backend.compile(
        package,
        selected,
        CompileOptions(mode=mode, dtype=dtype),
    )
    session = backend.load(artifact)
    engine = ExecutionEngine(
        package,
        session,
        batcher=adapter.collate,
        splitter=adapter.unbatch,
    )
    try:
        result = engine.infer(payload, num_steps=num_steps, seed=seed)
        actions = adapter.postprocess_one(result.output).actions
        return {
            "actions": actions,
            "shape": tuple(actions.shape),
            "model_id": result.metadata["model_id"],
            "backend": result.metadata["backend"],
            "device": result.metadata["device_id"],
            "execution_time_s": result.execution_time_s,
            "runtime_mode": session.actual_mode,
            "compile_failures": dict(session.compile_failures),
        }
    finally:
        engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the tiny staged-flow fixture through the local runtime."
    )
    parser.add_argument("--target", nargs="+", type=float, default=[0.25, -0.5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_toy_flow(
        target=args.target,
        device=args.device,
        dtype=args.dtype,
        mode=args.mode,
        num_steps=args.num_steps,
        seed=args.seed,
    )
    actions = summary.pop("actions")
    print("local toy-flow inference succeeded")
    for name, value in summary.items():
        print(f"{name}: {value}")
    print(f"actions:\n{actions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
