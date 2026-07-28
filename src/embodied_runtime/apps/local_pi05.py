"""Local real-weight pi0.5 smoke run with tokenizer-free synthetic inputs.

The checkpoint is loaded by the Group 1 adapter, execution is orchestrated by
the Group 3 engine, and all device/dtype handling goes through the Group 4
PyTorch backend.  A cached Hugging Face model id or a local snapshot directory
works without network access because loading is offline by default.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import CompileOptions
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.pi05 import Pi05Adapter


def _select_device(backend: TorchCudaBackend, device_id: str):
    devices = {device.device_id: device for device in backend.probe()}
    try:
        return devices[device_id]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"device {device_id!r} is unavailable; detected devices: {available}"
        ) from error


def run_pi05(
    *,
    checkpoint: str | Path,
    device: str = "cuda:0",
    dtype: str = "preserve",
    num_steps: int = 10,
    mode: str = "eager",
    batch_size: int = 1,
    language_length: int = 48,
    seed: int = 0,
    local_files_only: bool = True,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    """Load a verified checkpoint and execute one synthetic pi0.5 batch."""

    if cuda_graph and mode != "eager":
        raise ValueError("CUDA Graph currently requires --mode eager")
    if cuda_graph and not device.startswith("cuda:"):
        raise ValueError("CUDA Graph requires a cuda:N device")
    if batch_size != 1:
        raise ValueError(
            "the local synchronous π0.5 prototype accepts exactly one logical "
            "request; use separate requests plus a model-owned collator for batching"
        )
    adapter = Pi05Adapter()
    load_started = time.perf_counter()
    package = adapter.build_package(
        str(checkpoint),
        local_files_only=local_files_only,
    )
    load_time_s = time.perf_counter() - load_started

    # Group 1 creates already-tokenized synthetic tensors.  No tokenizer model
    # or network lookup is involved in this smoke path.
    payload = adapter.synthetic_batch(
        batch_size=batch_size,
        language_length=language_length,
        seed=seed,
    )

    backend = TorchCudaBackend()
    selected = _select_device(backend, device)
    backend_options: dict[str, Any] = {"empty_cache_on_close": True}
    if cuda_graph:
        # Capture only the repeated, RNG-free step. Initialization and the
        # engine-owned Euler update retain their ordinary execution semantics.
        backend_options["cuda_graph_entrypoints"] = (package.plan.step,)
        # The ten denoise calls share one graph.  Only the timestep value
        # changes, so the backend copies it into a fixed-address scalar buffer
        # before replay instead of treating each Python float as a new graph.
        backend_options["cuda_graph_dynamic_scalar_inputs"] = {
            package.plan.step: ("time",),
        }
    artifact = backend.compile(
        package,
        selected,
        CompileOptions(
            mode=mode,
            # The reference checkpoint deliberately keeps selected vision and
            # normalization paths in fp32.  Preserve that mixed policy unless a
            # user explicitly opts into a whole-module cast as an experiment.
            dtype=None if dtype == "preserve" else dtype,
            options=backend_options,
        ),
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
        actions = [adapter.postprocess_one(result.output).actions]
        return {
            "actions": actions,
            "action_shapes": [tuple(action.shape) for action in actions],
            "model_id": result.metadata["model_id"],
            "backend": result.metadata["backend"],
            "device": result.metadata["device_id"],
            "dtype": dtype,
            "num_steps": num_steps,
            "load_time_s": load_time_s,
            "execution_time_s": result.execution_time_s,
            "requested_mode": mode,
            "runtime_mode": session.actual_mode,
            "compile_failures": dict(session.compile_failures),
            "cuda_graph": cuda_graph,
            "cuda_graph_stats": dict(session.cuda_graph_stats),
            "cuda_graph_failures": dict(session.cuda_graph_failures),
            "offline": local_files_only,
        }
    finally:
        engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a verified pi0.5 checkpoint through the local model/engine/backend "
            "prototype with tokenizer-free synthetic input."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Local checkpoint directory/file or cached Hugging Face model id. "
            "The default loading policy is offline."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("preserve", "float32", "float16", "bfloat16"),
        default="preserve",
        help=(
            "Preserve the checkpoint's mixed dtypes (default), or experimentally "
            "cast the full module."
        ),
    )
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("eager", "compile"),
        default="eager",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Capture and replay the repeated denoise entrypoint on CUDA.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--language-length", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face network access when checkpoint files are not cached.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_steps <= 0:
        raise SystemExit("--num-steps must be greater than zero")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    if args.batch_size != 1:
        raise SystemExit(
            "--batch-size currently must be 1; dynamic batching requires separate "
            "logical requests and a model-owned collator"
        )
    if args.language_length <= 0:
        raise SystemExit("--language-length must be greater than zero")
    if args.cuda_graph and args.mode != "eager":
        raise SystemExit("--cuda-graph currently requires --mode eager")
    if args.cuda_graph and not args.device.startswith("cuda:"):
        raise SystemExit("--cuda-graph requires a cuda:N device")

    summary = run_pi05(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        num_steps=args.num_steps,
        mode=args.mode,
        batch_size=args.batch_size,
        language_length=args.language_length,
        seed=args.seed,
        local_files_only=not args.allow_download,
        cuda_graph=args.cuda_graph,
    )
    actions = summary.pop("actions")
    print("local pi0.5 inference succeeded")
    for name, value in summary.items():
        print(f"{name}: {value}")
    for index, action in enumerate(actions):
        preview = action.detach().float().cpu().reshape(-1)[:8]
        print(f"actions[{index}] preview: {preview.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
