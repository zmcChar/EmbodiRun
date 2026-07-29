"""GR00T N1.7 reference policy through an injected local hardware backend."""

from __future__ import annotations

from typing import Any

from embodied_runtime.backends import BackendRegistry, TorchCudaBackend
from embodied_runtime.contracts import CompileOptions, DeviceInfo
from embodied_runtime.engine import EngineConfig
from embodied_runtime.integrations.serving.local import LocalBackendProvider
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_CHECKPOINT,
    Gr00tN17Adapter,
)


class HfLocalGr00tProvider(LocalBackendProvider):
    """NVIDIA's reference GR00T policy exposed through a local Backend."""

    provider_name = "hf"
    provider_runtime = "nvidia_isaac_gr00t"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str = DEFAULT_CHECKPOINT,
        *,
        device: str | DeviceInfo = "cuda:0",
        mode: str = "eager",
        local_files_only: bool = True,
        cache_dir: str | None = None,
        revision: str | None = None,
        strict: bool = True,
        adapter: Gr00tN17Adapter | None = None,
        backends: BackendRegistry | None = None,
        backend_name: str | None = None,
        compile_options: CompileOptions | None = None,
        engine_config: EngineConfig | None = None,
        **package_options: Any,
    ) -> HfLocalGr00tProvider:
        selected_options = compile_options or CompileOptions(
            mode=mode,
            dtype=None,
            options={
                "offload_module_on_close": True,
                "empty_cache_on_close": True,
            },
        )
        if selected_options.mode != "eager":
            raise ValueError(
                "GR00T N1.7 currently supports only mode='eager'; compiling the "
                "full reference policy would incorrectly include NumPy preprocessing "
                "and CPU action decoding"
            )
        if compile_options is not None and mode != selected_options.mode:
            raise ValueError(
                f"mode={mode!r} conflicts with compile_options.mode={selected_options.mode!r}"
            )

        adapter = adapter or Gr00tN17Adapter()
        registry = backends or BackendRegistry([TorchCudaBackend()])
        selected_backend = backend_name
        if selected_backend is None and backends is None:
            selected_backend = "torch_cuda"

        options = {
            "local_files_only": local_files_only,
            "cache_dir": cache_dir,
            "revision": revision,
            "strict": strict,
            **package_options,
        }
        return super().from_checkpoint(
            adapter,
            checkpoint,
            backends=registry,
            device=device,
            backend_name=selected_backend,
            compile_options=selected_options,
            engine_config=engine_config,
            package_options=options,
        )


__all__ = ["HfLocalGr00tProvider"]
