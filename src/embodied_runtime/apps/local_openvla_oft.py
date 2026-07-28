"""Run a real OpenVLA-OFT checkpoint through the local runtime.

The Group 1 adapter owns image/text preprocessing and categorical action-token
semantics. Group 3 executes its ``SingleForwardPlan`` and Group 4 owns concrete
device placement. Checkpoint access is offline unless the caller explicitly
passes ``--allow-download``.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import CompileOptions, RawRequest
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models.vla.openvla_oft import OpenVLAOFTAdapter


def _select_device(backend: TorchCudaBackend, device_id: str):
    devices = {device.device_id: device for device in backend.probe()}
    try:
        return devices[device_id]
    except KeyError as error:
        available = ", ".join(devices) or "<none>"
        raise RuntimeError(
            f"device {device_id!r} is unavailable; detected devices: {available}"
        ) from error


def run_openvla_oft(
    *,
    checkpoint: str | Path,
    prompt: str = "pick up the red block",
    device: str = "cuda:0",
    dtype: str = "preserve",
    load_dtype: str = "bfloat16",
    mode: str = "eager",
    image_size: int = 224,
    seed: int = 0,
    local_files_only: bool = True,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Load one checkpoint and execute one synthetic-image request."""

    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    # Import lazily enough that listing adapters and reading the CLI help do
    # not allocate a model or consult a checkpoint.
    import torch

    adapter = OpenVLAOFTAdapter()
    load_started = time.perf_counter()
    package = adapter.build_package(
        str(checkpoint),
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        revision=revision,
        load_dtype=load_dtype,
    )
    load_time_s = time.perf_counter() - load_started

    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = torch.rand(
        (3, image_size, image_size),
        dtype=torch.float32,
        generator=generator,
    )
    payload = adapter.preprocess_one(
        RawRequest(
            observation={"image": image},
            prompt=prompt,
            metadata={"synthetic": True},
        )
    )

    backend = TorchCudaBackend()
    selected = _select_device(backend, device)
    artifact = backend.compile(
        package,
        selected,
        CompileOptions(
            mode=mode,
            dtype=None if dtype == "preserve" else dtype,
            options={"empty_cache_on_close": True},
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
        result = engine.infer(payload)
        actions = adapter.postprocess_one(result.output).actions
        return {
            "actions": actions,
            "action_shape": tuple(actions.shape),
            "model_id": result.metadata["model_id"],
            "plan_kind": package.plan.kind,
            "backend": result.metadata["backend"],
            "device": result.metadata["device_id"],
            "dtype": dtype,
            "load_dtype": load_dtype,
            "load_time_s": load_time_s,
            "execution_time_s": result.execution_time_s,
            "requested_mode": mode,
            "runtime_mode": session.actual_mode,
            "compile_failures": dict(session.compile_failures),
            "offline": local_files_only,
        }
    finally:
        engine.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenVLA-OFT checkpoint through the local "
            "model/engine/backend prototype with a synthetic RGB image."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Local checkpoint directory or cached Hugging Face model id. "
            "The default loading policy is offline."
        ),
    )
    parser.add_argument("--prompt", default="pick up the red block")
    parser.add_argument("--device", default="cuda:0", help="cpu or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=("preserve", "float32", "float16", "bfloat16"),
        default="preserve",
        help="Preserve checkpoint dtypes (default), or cast the complete runtime module.",
    )
    parser.add_argument(
        "--load-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Floating dtype used while constructing and loading the checkpoint.",
    )
    parser.add_argument("--mode", choices=("eager", "compile"), default="eager")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--revision")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face network access when checkpoint files are not cached.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size <= 0:
        raise SystemExit("--image-size must be greater than zero")

    summary = run_openvla_oft(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        device=args.device,
        dtype=args.dtype,
        load_dtype=args.load_dtype,
        mode=args.mode,
        image_size=args.image_size,
        seed=args.seed,
        local_files_only=not args.allow_download,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    actions = summary.pop("actions")
    print("local OpenVLA-OFT inference succeeded")
    for name, value in summary.items():
        print(f"{name}: {value}")
    preview = actions.detach().float().cpu().reshape(-1)[:8]
    print(f"actions preview: {preview.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
