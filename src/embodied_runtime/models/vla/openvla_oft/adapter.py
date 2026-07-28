"""ModelAdapter for the RLinf-compatible OpenVLA-OFT checkpoint family."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from embodied_runtime.contracts import (
    EntrypointSpec,
    ModelPackage,
    ModelPackageError,
    ModelSpec,
    RawRequest,
    SingleForwardPlan,
)

from ...registry import register_model
from ..base import VLAAdapterBase
from .modeling_openvla_oft import load_openvla_oft

_VVLA_SOURCE_COMMIT = "80b5cf48c8710c69ed97200903562e9787efe105"


@register_model("openvla_oft")
class OpenVLAOFTAdapter(VLAAdapterBase):
    """Package OpenVLA-OFT as one backend-neutral single-forward plan."""

    def __init__(self) -> None:
        self._processor: Any | None = None
        self._module: torch.nn.Module | None = None
        self._spec: ModelSpec | None = None

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise ModelPackageError("build_package must be called before preprocessing")
        return self._processor

    @property
    def runtime_module(self) -> torch.nn.Module:
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
        """Load a real checkpoint or inject tiny components for contract tests.

        ``module=...`` and ``processor=...`` must be supplied together.  The
        injection seam is deliberately above the engine/backend boundary: it
        tests the same ModelPackage and SingleForwardPlan without allocating an
        8B checkpoint.
        """

        module = options.pop("module", None)
        processor = options.pop("processor", None)
        requested_action_dim = options.pop("action_dim", None)
        requested_horizon = options.pop("action_horizon", None)
        revision = options.pop("revision", None)
        loader_metadata: dict[str, Any]

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
            if not isinstance(module, torch.nn.Module):
                raise TypeError("OpenVLA-OFT runtime module must be torch.nn.Module")
            loader_metadata = {"injected": True}

        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"unknown OpenVLA-OFT package option(s): {unknown}")
        assert processor is not None
        assert module is not None
        action_dim = int(
            requested_action_dim
            if requested_action_dim is not None
            else getattr(module, "action_dim", 7)
        )
        action_horizon = int(
            requested_horizon
            if requested_horizon is not None
            else getattr(module, "action_horizon", 8)
        )
        if action_dim <= 0 or action_horizon <= 0:
            raise ModelPackageError(
                "OpenVLA-OFT action_dim and action_horizon must be greater than zero"
            )

        self._module = module.eval()
        self._processor = processor
        self._spec = ModelSpec(
            model_id="openvla-oft",
            family="openvla_oft_categorical",
            revision=revision,
            modalities=("vision", "language"),
            action_dim=action_dim,
            action_horizon=action_horizon,
            metadata={
                "framework": "torch",
                "generation": "categorical_greedy",
                "execution_plan": "single_forward",
                "checkpoint": checkpoint,
            },
        )
        plan = SingleForwardPlan()
        return ModelPackage(
            spec=self._spec,
            checkpoint=checkpoint or None,
            plan=plan,
            entrypoints={plan.forward: self._module.forward},
            entrypoint_specs={
                plan.forward: EntrypointSpec(
                    plan.forward,
                    "Run vision, multimodal Llama, and greedy categorical action decoding.",
                    batchable=True,
                    safe_point_after=True,
                    metadata={
                        "generation": "categorical_greedy",
                        "output": "action_chunk",
                    },
                )
            },
            metadata={
                "framework": "torch",
                "runtime_module": self._module,
                "portable": False,
                "generation": "categorical_greedy",
                "loader": loader_metadata,
                "source": {
                    "vvla_commit": _VVLA_SOURCE_COMMIT,
                    "vvla_license": "MIT",
                    "rlinf_reference_license": "Apache-2.0",
                    "openvla_oft_checkpoint_license": "MIT",
                },
                "limitations": {
                    "kv_split": False,
                    "rl_logprob_recompute": False,
                    "cuda_graph_validated": False,
                },
            },
        )

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
