"""Configuration helper shared by multi-robot composition roots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from embodied_runtime.engine.config import EngineConfig

if TYPE_CHECKING:
    from embodied_runtime.engine.providers import LocalBackendProvider


@dataclass(frozen=True, slots=True)
class LocalProviderConfig:
    model: str = "toy_single_forward"
    checkpoint: str = ""
    backend: str = "torch_cuda"
    device: str = "cpu"
    provider_name: str = "hf-local"
    provider_runtime: str = "local_backend"
    multi_tenant_safe: bool = False
    compile_mode: str = "eager"
    dtype: str | None = "float32"
    dynamic_shapes: bool = False
    adapter_options: Mapping[str, Any] = field(default_factory=dict)
    package_options: Mapping[str, Any] = field(default_factory=dict)
    compile_options: Mapping[str, Any] = field(default_factory=dict)
    max_queue_size: int = 128
    max_batch_size: int = 1
    max_wait_ms: float = 0.0
    memory_budget_bytes: int | None = None
    minimum_free_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("provider model must not be empty")
        if not self.backend.strip():
            raise ValueError("provider backend must not be empty")
        if not self.device.strip():
            raise ValueError("provider device must not be empty")
        if not isinstance(self.multi_tenant_safe, bool):
            raise TypeError("provider multi_tenant_safe must be a boolean")
        if not isinstance(self.dynamic_shapes, bool):
            raise TypeError("provider dynamic_shapes must be a boolean")
        if self.dtype == "preserve":
            object.__setattr__(self, "dtype", None)
        # Reuse the engine's validation without retaining another object.
        self.engine_config()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> LocalProviderConfig:
        engine = values.get("engine", {})
        if not isinstance(engine, Mapping):
            raise TypeError("provider.engine must be a table")
        adapter_options = values.get("adapter_options", {})
        package_options = values.get("package_options", {})
        compile_options = values.get("options", {})
        for name, option_values in (
            ("adapter_options", adapter_options),
            ("package_options", package_options),
            ("options", compile_options),
        ):
            if not isinstance(option_values, Mapping):
                raise TypeError(f"provider.{name} must be a table")
        multi_tenant_safe = values.get("multi_tenant_safe", False)
        dynamic_shapes = values.get("dynamic_shapes", False)
        if not isinstance(multi_tenant_safe, bool):
            raise TypeError("provider.multi_tenant_safe must be a boolean")
        if not isinstance(dynamic_shapes, bool):
            raise TypeError("provider.dynamic_shapes must be a boolean")
        dtype = values.get("dtype", "float32")
        return cls(
            model=str(values.get("model", "toy_single_forward")),
            checkpoint=str(values.get("checkpoint", "")),
            backend=str(values.get("backend", "torch_cuda")),
            device=str(values.get("device", "cpu")),
            provider_name=str(values.get("name", "hf-local")),
            provider_runtime=str(values.get("runtime", "local_backend")),
            multi_tenant_safe=multi_tenant_safe,
            compile_mode=str(values.get("compile_mode", "eager")),
            dtype=None if dtype is None else str(dtype),
            dynamic_shapes=dynamic_shapes,
            adapter_options=dict(adapter_options),
            package_options=dict(package_options),
            compile_options=dict(compile_options),
            max_queue_size=int(engine.get("max_queue_size", 128)),
            max_batch_size=int(engine.get("max_batch_size", 1)),
            max_wait_ms=float(engine.get("max_wait_ms", 0.0)),
            memory_budget_bytes=(
                None
                if engine.get("memory_budget_bytes") is None
                else int(engine["memory_budget_bytes"])
            ),
            minimum_free_bytes=int(engine.get("minimum_free_bytes", 0)),
        )

    def engine_config(self) -> EngineConfig:
        return EngineConfig(
            max_queue_size=self.max_queue_size,
            max_batch_size=self.max_batch_size,
            max_wait_ms=self.max_wait_ms,
            memory_budget_bytes=self.memory_budget_bytes,
            minimum_free_bytes=self.minimum_free_bytes,
        )


def build_local_provider(config: LocalProviderConfig) -> LocalBackendProvider:
    """Build one local Provider without hiding model or device placement."""

    # Concrete backends and model adapters stay behind the runtime construction
    # boundary so config parsing works in installations without PyTorch/model extras.
    from embodied_runtime.backends.compile import CompileOptions
    from embodied_runtime.backends.registry import BackendRegistry
    from embodied_runtime.backends.torch_cuda import TorchCudaBackend
    from embodied_runtime.engine.providers import LocalBackendProvider
    from embodied_runtime.models.registry import get_model_adapter

    if config.backend != "torch_cuda":
        raise ValueError(
            f"backend {config.backend!r} has no app-level factory yet; "
            "inject it through BackendRegistry in a deployment composition"
        )
    adapter = get_model_adapter(
        config.model,
        **dict(config.adapter_options),
    )
    registry = BackendRegistry([TorchCudaBackend()])
    return LocalBackendProvider.from_checkpoint(
        adapter,
        config.checkpoint,
        backends=registry,
        backend_name=config.backend,
        device=config.device,
        compile_options=CompileOptions(
            mode=config.compile_mode,
            dtype=config.dtype,
            dynamic_shapes=config.dynamic_shapes,
            options=dict(config.compile_options),
        ),
        engine_config=config.engine_config(),
        package_options=dict(config.package_options),
        provider_name=config.provider_name,
        provider_runtime=config.provider_runtime,
        provider_features=(
            frozenset({"multi_tenant_safe"}) if config.multi_tenant_safe else frozenset()
        ),
    )


__all__ = ["LocalProviderConfig", "build_local_provider"]
