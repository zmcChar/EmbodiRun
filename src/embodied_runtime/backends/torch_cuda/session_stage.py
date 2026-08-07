"""Entrypoint selection, fallback, and synchronized stage execution."""

from __future__ import annotations

from typing import Any

from embodied_runtime.engine.errors import RequestCancelledError

from ..errors import BackendExecutionError
from .compiler import TorchArtifactPayload
from .cuda_graph import CudaGraphExecutor


def actual_execution_mode(payload: TorchArtifactPayload) -> str:
    modes = {
        payload.entrypoints[name] is payload.eager_entrypoints[name] for name in payload.entrypoints
    }
    if modes == {True}:
        return "eager"
    if modes == {False}:
        return "compile"
    return "mixed"


def invoke_entrypoint(
    payload: TorchArtifactPayload,
    entrypoint: str,
    inputs: Any,
) -> Any:
    try:
        function = payload.entrypoints[entrypoint]
    except KeyError as error:
        available = ", ".join(sorted(payload.entrypoints))
        raise BackendExecutionError(
            f"unknown model entrypoint {entrypoint!r}; available: {available}"
        ) from error

    try:
        return function(inputs)
    except Exception as compiled_error:
        eager = payload.eager_entrypoints[entrypoint]
        is_compiled = function is not eager
        if not (is_compiled and payload.fallback_to_eager):
            raise BackendExecutionError(
                f"PyTorch entrypoint {entrypoint!r} failed on "
                f"{payload.device_id}: {type(compiled_error).__name__}: "
                f"{compiled_error}"
            ) from compiled_error
        try:
            output = eager(inputs)
        except Exception as eager_error:
            raise BackendExecutionError(
                f"compiled and eager execution both failed for {entrypoint!r}; "
                f"compiled={type(compiled_error).__name__}: {compiled_error}; "
                f"eager={type(eager_error).__name__}: {eager_error}"
            ) from eager_error
        payload.entrypoints[entrypoint] = eager
        payload.compile_failures[entrypoint] = (
            f"deferred {type(compiled_error).__name__}: {compiled_error}"
        )
        return output


def synchronize_if_needed(
    torch: Any,
    torch_device: Any,
    payload: TorchArtifactPayload,
) -> None:
    if payload.synchronize_on_submit and payload.device_id.startswith("cuda"):
        torch.cuda.synchronize(torch_device)


def execute_stage(
    torch: Any,
    torch_device: Any,
    payload: TorchArtifactPayload,
    cuda_graphs: CudaGraphExecutor | None,
    entrypoint: str,
    moved_inputs: Any,
    *,
    use_cuda_graph: bool,
) -> Any:
    try:
        if use_cuda_graph:
            assert cuda_graphs is not None

            def invoke(inputs: Any) -> Any:
                return invoke_entrypoint(payload, entrypoint, inputs)

            if payload.inference_mode:
                with torch.inference_mode():
                    output = cuda_graphs.execute(entrypoint, moved_inputs, invoke)
            else:  # rejected when CUDA Graph entrypoints are configured
                output = cuda_graphs.execute(entrypoint, moved_inputs, invoke)
        elif payload.inference_mode:
            with torch.inference_mode():
                output = invoke_entrypoint(payload, entrypoint, moved_inputs)
        else:
            output = invoke_entrypoint(payload, entrypoint, moved_inputs)
    except (BackendExecutionError, RequestCancelledError):
        raise
    except Exception as error:
        raise BackendExecutionError(
            f"PyTorch entrypoint {entrypoint!r} failed: {type(error).__name__}: {error}"
        ) from error
    try:
        synchronize_if_needed(torch, torch_device, payload)
    except Exception as error:
        raise BackendExecutionError(
            f"CUDA synchronization failed after entrypoint {entrypoint!r}: "
            f"{type(error).__name__}: {error}"
        ) from error
    return output
