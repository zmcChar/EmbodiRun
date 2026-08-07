"""Build reusable PyTorch artifact payloads from validated selections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.models.package import ModelPackage

from ..compile import CompileOptions
from .compiler_entrypoints import compile_entrypoints
from .compiler_validation import (
    normalize_mode,
    resolve_cuda_graph_config,
    resolve_dtype,
)
from .cuda_graph import CudaGraphConfig


@dataclass(slots=True)
class TorchArtifactPayload:
    """Device-specialized callables and their eager fallback counterparts."""

    package_id: str
    entrypoints: dict[str, Callable[[Any], Any]]
    eager_entrypoints: dict[str, Callable[[Any], Any]]
    device_id: str
    dtype: Any | None
    requested_mode: str
    actual_mode: str
    runtime_module: Any | None = None
    fallback_to_eager: bool = False
    inference_mode: bool = True
    non_blocking: bool = False
    synchronize_on_submit: bool = True
    offload_module_on_close: bool = False
    empty_cache_on_close: bool = False
    compile_failures: dict[str, str] = field(default_factory=dict)
    cuda_graph: CudaGraphConfig = field(default_factory=CudaGraphConfig)

    def for_session(self) -> TorchArtifactPayload:
        """Return a private mutable runtime view of this reusable artifact."""

        return TorchArtifactPayload(
            package_id=self.package_id,
            entrypoints=dict(self.entrypoints),
            eager_entrypoints=dict(self.eager_entrypoints),
            device_id=self.device_id,
            dtype=self.dtype,
            requested_mode=self.requested_mode,
            actual_mode=self.actual_mode,
            runtime_module=self.runtime_module,
            fallback_to_eager=self.fallback_to_eager,
            inference_mode=self.inference_mode,
            non_blocking=self.non_blocking,
            synchronize_on_submit=self.synchronize_on_submit,
            offload_module_on_close=self.offload_module_on_close,
            empty_cache_on_close=self.empty_cache_on_close,
            compile_failures=dict(self.compile_failures),
            cuda_graph=self.cuda_graph,
        )


def get_runtime_module(
    torch: Any,
    package: ModelPackage,
) -> Any | None:
    """Validate and return the optional package-owned module without moving it."""

    module = package.metadata.get("runtime_module")
    if module is None:
        return None
    if not isinstance(module, torch.nn.Module):
        raise TypeError("ModelPackage.metadata['runtime_module'] must be torch.nn.Module")
    return module


def move_runtime_module(
    torch: Any,
    module: Any | None,
    *,
    device_id: str,
    dtype: Any | None,
) -> Any | None:
    """Place a leased runtime module on the session's concrete device."""

    if module is None:
        return None
    if dtype is None:
        module.to(device=torch.device(device_id))
    else:
        module.to(device=torch.device(device_id), dtype=dtype)
    module.eval()
    return module


def make_payload(
    torch: Any,
    package: ModelPackage,
    options: CompileOptions,
    *,
    device_id: str,
) -> TorchArtifactPayload:
    dtype = resolve_dtype(torch, options.dtype)
    runtime_module = get_runtime_module(torch, package)
    cuda_graph = resolve_cuda_graph_config(
        torch,
        package,
        options,
        device_id=device_id,
    )
    entrypoints, eager, failures, actual_mode = compile_entrypoints(torch, package, options)
    is_cuda = device_id.startswith("cuda")
    return TorchArtifactPayload(
        package_id=package.package_id,
        entrypoints=entrypoints,
        eager_entrypoints=eager,
        device_id=device_id,
        dtype=dtype,
        requested_mode=normalize_mode(options.mode),
        actual_mode=actual_mode,
        runtime_module=runtime_module,
        fallback_to_eager=bool(options.options.get("fallback_to_eager", False)),
        inference_mode=bool(options.options.get("inference_mode", True)),
        non_blocking=bool(options.options.get("non_blocking", False)),
        synchronize_on_submit=bool(options.options.get("synchronize_on_submit", True)),
        offload_module_on_close=bool(options.options.get("offload_module_on_close", is_cuda)),
        empty_cache_on_close=bool(options.options.get("empty_cache_on_close", is_cuda)),
        compile_failures=failures,
        cuda_graph=cuda_graph,
    )


__all__ = [
    "TorchArtifactPayload",
    "compile_entrypoints",
    "get_runtime_module",
    "make_payload",
    "move_runtime_module",
    "normalize_mode",
    "resolve_cuda_graph_config",
    "resolve_dtype",
]
