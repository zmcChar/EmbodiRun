"""Compilation helpers kept separate from model-family code."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.contracts import CompileOptions, ModelPackage, UnsupportedBackendError

from .cuda_graph import CudaGraphConfig


_EAGER_MODES = frozenset({"eager", "torch_eager"})
_COMPILE_MODES = frozenset({"compile", "torch.compile", "torch_compile"})


@dataclass(slots=True)
class TorchArtifactPayload:
    """In-memory payload produced by :class:`TorchCudaBackend`.

    The portable ``ModelPackage`` remains the source of truth.  This payload
    records the device-specialized callables and their eager counterparts so a
    failed lazy ``torch.compile`` specialization can safely fall back.
    """

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


def compile_entrypoints(
    torch: Any,
    package: ModelPackage,
    options: CompileOptions,
) -> tuple[
    dict[str, Callable[[Any], Any]],
    dict[str, Callable[[Any], Any]],
    dict[str, str],
    str,
]:
    """Create eager or lazily compiled stage callables.

    ``torch.compile`` normally specializes on first invocation, so both
    immediate wrapping failures and deferred execution failures are supported.
    Deferred fallback is implemented by ``TorchBackendSession``.
    """

    eager = dict(package.entrypoints)
    mode = normalize_mode(options.mode)
    if mode == "eager":
        return dict(eager), eager, {}, "eager"

    fallback = bool(options.options.get("fallback_to_eager", False))
    if not hasattr(torch, "compile"):
        if fallback:
            return (
                dict(eager),
                eager,
                {"*": "torch.compile is unavailable in this PyTorch build"},
                "eager",
            )
        raise UnsupportedBackendError("torch.compile is unavailable in this PyTorch build")

    compile_kwargs: dict[str, Any] = {
        "fullgraph": bool(options.options.get("fullgraph", False)),
        "dynamic": bool(options.dynamic_shapes),
    }
    compiler_backend = options.options.get("compiler_backend")
    if compiler_backend is not None:
        compile_kwargs["backend"] = compiler_backend
    compiler_mode = options.options.get("compiler_mode")
    if compiler_mode is not None:
        compile_kwargs["mode"] = compiler_mode

    compiled: dict[str, Callable[[Any], Any]] = {}
    failures: dict[str, str] = {}
    for name, entrypoint in eager.items():
        try:
            compiled[name] = torch.compile(entrypoint, **compile_kwargs)
        except Exception as error:
            if not fallback:
                raise UnsupportedBackendError(
                    f"torch.compile could not wrap entrypoint {name!r}: "
                    f"{type(error).__name__}: {error}"
                ) from error
            compiled[name] = entrypoint
            failures[name] = f"{type(error).__name__}: {error}"

    if not failures:
        actual_mode = "compile"
    elif len(failures) == len(eager):
        actual_mode = "eager"
    else:
        actual_mode = "mixed"
    return compiled, eager, failures, actual_mode


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
