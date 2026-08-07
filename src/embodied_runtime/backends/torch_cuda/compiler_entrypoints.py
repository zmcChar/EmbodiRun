"""Select eager or compiled callables for a PyTorch artifact."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from embodied_runtime.models.package import ModelPackage

from ..compile import CompileOptions
from ..errors import UnsupportedBackendError
from .compiler_validation import normalize_mode


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
    """Create eager or lazily compiled stage callables."""

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


__all__ = ["compile_entrypoints"]
