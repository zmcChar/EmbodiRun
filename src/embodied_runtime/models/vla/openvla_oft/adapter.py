"""Lightweight orchestration for the OpenVLA-OFT adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...registry import register_model
from ...request import RawRequest
from ...spec import ModelSpec
from ..base import VLAAdapterBase
from .loading import load_openvla_oft


@register_model("openvla_oft")
class OpenVLAOFTAdapter(VLAAdapterBase):
    """Package OpenVLA-OFT as one backend-neutral single-forward plan."""

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._module: Any | None = None
        self._spec: ModelSpec | None = None

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise ModelPackageError("build_package must be called before preprocessing")
        return self._processor

    @property
    def runtime_module(self) -> Any:
        if self._module is None:
            raise ModelPackageError("build_package must be called before accessing the module")
        return self._module

    def describe(self) -> ModelSpec:
        if self._spec is not None:
            return self._spec
        return ModelSpec(
            model_id="openvla-oft",
            family="openvla_oft_categorical",
            modalities=("vision", "language"),
            action_dim=7,
            action_horizon=8,
            metadata={
                "framework": "torch",
                "generation": "categorical_greedy",
                "execution_plan": "single_forward",
            },
        )

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage:
        module = options.pop("module", None)
        processor = options.pop("processor", None)
        requested_action_dim = options.pop("action_dim", None)
        requested_horizon = options.pop("action_horizon", None)
        revision = options.pop("revision", None)

        if (module is None) != (processor is None):
            raise ModelPackageError("OpenVLA-OFT test injection requires both module and processor")
        if module is None:
            module, processor, loader_metadata = load_openvla_oft(
                checkpoint,
                load_device=options.pop("load_device", "cpu"),
                load_dtype=options.pop("load_dtype", "bfloat16"),
                cache_dir=options.pop("cache_dir", None),
                revision=revision,
                local_files_only=options.pop("local_files_only", False),
                strict=options.pop("strict", True),
                action_dim=requested_action_dim,
                action_horizon=requested_horizon,
                max_prompt_length=options.pop("max_prompt_length", 50),
            )
        else:
            loader_only = {
                "load_device",
                "load_dtype",
                "cache_dir",
                "local_files_only",
                "strict",
                "max_prompt_length",
            }.intersection(options)
            if loader_only:
                names = ", ".join(sorted(loader_only))
                raise TypeError(
                    f"injected OpenVLA-OFT components do not accept loader option(s): {names}"
                )
            from .dependencies import require_torch

            if not isinstance(module, require_torch().nn.Module):
                raise TypeError("OpenVLA-OFT runtime module must be torch.nn.Module")
            loader_metadata = {"injected": True}

        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"unknown OpenVLA-OFT package option(s): {unknown}")
        assert processor is not None
        assert module is not None

        from .package import build_openvla_oft_package

        components = build_openvla_oft_package(
            module,
            processor,
            checkpoint=checkpoint,
            revision=revision,
            requested_action_dim=requested_action_dim,
            requested_action_horizon=requested_horizon,
            loader_metadata=loader_metadata,
        )
        self._module = components.module
        self._processor = components.processor
        self._spec = components.spec
        return components.package

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        preprocess = getattr(self.processor, "preprocess_one", None)
        if not callable(preprocess):
            raise ModelPackageError(
                "OpenVLA-OFT processor must implement preprocess_one(RawRequest)"
            )
        payload = preprocess(request)
        if not isinstance(payload, Mapping):
            raise ModelPackageError("OpenVLA-OFT processor preprocess_one must return a mapping")
        return payload


__all__ = ["OpenVLAOFTAdapter"]
