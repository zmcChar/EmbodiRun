"""Model-adapter registry owned by the model integration group.

The registry stores *adapter factories*, not loaded model instances.  Loading a
checkpoint remains an explicit ``ModelAdapter.build_package`` operation, so
enumerating available model families never allocates weights or imports a
hardware backend.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from .errors import ModelPackageError
from .interfaces import ModelAdapter

AdapterFactory = Callable[..., ModelAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}
_BUILTIN_MODULES: dict[str, str] = {
    "gr00t_n17": "embodied_runtime.models.vla.gr00t_n17",
    "openvla_oft": "embodied_runtime.models.vla.openvla_oft",
    "pi05": "embodied_runtime.models.vla.pi05",
    "smolvla": "embodied_runtime.models.vla.smolvla",
    "toy_flow": "embodied_runtime.models.vla.toy_flow",
    "toy_single_forward": "embodied_runtime.models.vla.toy_single_forward",
}


def register_model(
    name: str,
    factory: AdapterFactory | None = None,
    *,
    replace: bool = False,
) -> AdapterFactory | Callable[[AdapterFactory], AdapterFactory]:
    """Register an adapter factory.

    It can be used directly or as a decorator::

        register_model("my_model", MyAdapter)

        @register_model("another_model")
        class AnotherAdapter: ...
    """

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("model name must not be empty")

    def install(candidate: AdapterFactory) -> AdapterFactory:
        if normalized in _REGISTRY and not replace:
            raise ValueError(f"model adapter {normalized!r} is already registered")
        _REGISTRY[normalized] = candidate
        return candidate

    if factory is None:
        return install
    return install(factory)


def _load_builtin(name: str) -> None:
    module_name = _BUILTIN_MODULES.get(name)
    if module_name is not None and name not in _REGISTRY:
        importlib.import_module(module_name)


def get_model_adapter(name: str, **options: Any) -> ModelAdapter:
    """Construct a registered adapter without loading its checkpoint."""

    normalized = name.strip().lower()
    _load_builtin(normalized)
    try:
        factory = _REGISTRY[normalized]
    except KeyError as exc:
        available = ", ".join(available_models()) or "<none>"
        raise ModelPackageError(
            f"unknown model adapter {name!r}; available adapters: {available}"
        ) from exc
    adapter = factory(**options)
    if not isinstance(adapter, ModelAdapter):
        raise TypeError(
            f"registered factory for {normalized!r} returned "
            f"{type(adapter).__name__}, which does not satisfy ModelAdapter"
        )
    return adapter


def available_models() -> tuple[str, ...]:
    """List registered and lazily available built-in model families."""

    return tuple(sorted(set(_BUILTIN_MODULES).union(_REGISTRY)))


__all__ = [
    "AdapterFactory",
    "available_models",
    "get_model_adapter",
    "register_model",
]
