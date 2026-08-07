"""Validation and normalization for PyTorch compilation options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_runtime.models.package import ModelPackage

from ..compile import CompileOptions
from ..errors import UnsupportedBackendError
from .cuda_graph import CudaGraphConfig

_EAGER_MODES = frozenset({"eager", "torch_eager"})
_COMPILE_MODES = frozenset({"compile", "torch.compile", "torch_compile"})


def normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized in _EAGER_MODES:
        return "eager"
    if normalized in _COMPILE_MODES:
        return "compile"
    choices = ", ".join(sorted(_EAGER_MODES | _COMPILE_MODES))
    raise UnsupportedBackendError(
        f"unsupported PyTorch execution mode {mode!r}; expected one of: {choices}"
    )


def resolve_dtype(torch: Any, dtype: str | None) -> Any | None:
    if dtype is None:
        return None
    normalized = dtype.lower().replace("torch.", "").replace("_", "")
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "double": torch.float64,
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        supported = ", ".join(sorted(aliases))
        raise UnsupportedBackendError(
            f"unsupported PyTorch dtype {dtype!r}; known aliases: {supported}"
        ) from error


def resolve_cuda_graph_config(
    torch: Any,
    package: ModelPackage,
    options: CompileOptions,
    *,
    device_id: str,
) -> CudaGraphConfig:
    """Validate the small, backend-private CUDA Graph option surface."""

    raw_entrypoints = options.options.get("cuda_graph_entrypoints", ())
    if isinstance(raw_entrypoints, str):
        entrypoints = (raw_entrypoints,)
    else:
        try:
            entrypoints = tuple(raw_entrypoints)
        except TypeError as error:
            raise UnsupportedBackendError(
                "cuda_graph_entrypoints must be a string or an iterable of strings"
            ) from error
    if any(not isinstance(name, str) or not name for name in entrypoints):
        raise UnsupportedBackendError("cuda_graph_entrypoints must contain non-empty strings")
    selected = frozenset(entrypoints)

    raw_dynamic_scalars = options.options.get("cuda_graph_dynamic_scalar_inputs", {})
    if not isinstance(raw_dynamic_scalars, Mapping):
        raise UnsupportedBackendError(
            "cuda_graph_dynamic_scalar_inputs must map entrypoints to input names"
        )
    dynamic_scalar_inputs: dict[str, frozenset[str]] = {}
    for entrypoint, raw_names in raw_dynamic_scalars.items():
        if not isinstance(entrypoint, str) or not entrypoint:
            raise UnsupportedBackendError(
                "cuda_graph_dynamic_scalar_inputs must use non-empty entrypoint names"
            )
        if isinstance(raw_names, str):
            names = (raw_names,)
        else:
            try:
                names = tuple(raw_names)
            except TypeError as error:
                raise UnsupportedBackendError(
                    "cuda_graph_dynamic_scalar_inputs values must be strings "
                    "or iterables of strings"
                ) from error
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise UnsupportedBackendError(
                "cuda_graph_dynamic_scalar_inputs values must contain non-empty input names"
            )
        dynamic_scalar_inputs[entrypoint] = frozenset(names)

    warmup_steps = options.options.get("cuda_graph_warmup_steps", 1)
    max_graphs = options.options.get("cuda_graph_max_graphs", 16)
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise UnsupportedBackendError("cuda_graph_warmup_steps must be a positive integer")
    if isinstance(max_graphs, bool) or not isinstance(max_graphs, int) or max_graphs < 1:
        raise UnsupportedBackendError("cuda_graph_max_graphs must be a positive integer")
    unselected_dynamic = set(dynamic_scalar_inputs).difference(selected)
    if unselected_dynamic:
        names = ", ".join(sorted(unselected_dynamic))
        raise UnsupportedBackendError(
            f"dynamic scalar inputs require selected CUDA Graph entrypoint(s): {names}"
        )
    if not selected:
        return CudaGraphConfig(
            warmup_steps=warmup_steps,
            max_graphs=max_graphs,
        )

    unknown = selected.difference(package.entrypoints)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise UnsupportedBackendError(f"unknown CUDA Graph entrypoint(s): {names}")
    if not device_id.startswith("cuda:"):
        raise UnsupportedBackendError("CUDA Graph entrypoints require a CUDA device")
    if normalize_mode(options.mode) != "eager":
        raise UnsupportedBackendError(
            "explicit CUDA Graph entrypoints currently require PyTorch eager mode"
        )
    if options.dynamic_shapes:
        raise UnsupportedBackendError(
            "explicit CUDA Graph entrypoints require dynamic_shapes=False"
        )
    if not bool(options.options.get("inference_mode", True)):
        raise UnsupportedBackendError("explicit CUDA Graph entrypoints require inference_mode=True")
    if not bool(options.options.get("synchronize_on_submit", True)):
        raise UnsupportedBackendError(
            "explicit CUDA Graph entrypoints require synchronize_on_submit=True"
        )
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph"):
        raise UnsupportedBackendError("this PyTorch build does not expose CUDA Graph APIs")
    return CudaGraphConfig(
        entrypoints=selected,
        dynamic_scalar_inputs=dynamic_scalar_inputs,
        warmup_steps=warmup_steps,
        max_graphs=max_graphs,
    )


__all__ = ["normalize_mode", "resolve_cuda_graph_config", "resolve_dtype"]
