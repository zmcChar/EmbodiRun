"""Lightweight SmolVLA adapter orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...registry import register_model
from ...request import RawRequest
from ...spec import ModelSpec
from ..base import VLAAdapterBase
from .checkpoint import load_lerobot_smolvla
from .constants import DEFAULT_SMOLVLA_REVISION, EXPECTED_LEROBOT_VERSION


@register_model("smolvla")
class SmolVLAAdapter(VLAAdapterBase):
    """Build a staged flow package around a verified SmolVLA policy."""

    def __init__(self) -> None:
        self._policy: Any | None = None
        self._processor: Any | None = None
        self._module: Any | None = None
        self._spec: ModelSpec | None = None

    @property
    def loaded_policy(self) -> Any:
        if self._policy is None:
            raise ModelPackageError("build_package must be called before accessing the policy")
        return self._policy

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise ModelPackageError("build_package must be called before preprocessing")
        return self._processor

    def describe(self) -> ModelSpec:
        if self._spec is not None:
            return self._spec
        return ModelSpec(
            model_id="smolvla",
            family="smolvla_flow",
            modalities=("vision", "language", "proprioception"),
            action_dim=6,
            action_horizon=50,
            metadata={
                "framework": "torch",
                "reference_loader": f"lerobot=={EXPECTED_LEROBOT_VERSION}",
                "execution_plan": "iterative_flow",
                "internal_action_dim": 32,
            },
        )

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage:
        policy = options.pop("policy", None)
        revision = options.pop("revision", DEFAULT_SMOLVLA_REVISION)
        stats_variant = options.pop("stats_variant", "so100")
        default_num_steps = options.pop("default_num_steps", None)
        if policy is None:
            vlm_base_path = options.pop("vlm_base_path", None)
            if vlm_base_path is None:
                raise ModelPackageError(
                    "SmolVLA package loading requires vlm_base_path for an explicit "
                    "offline SmolVLM2 dependency"
                )
            policy = load_lerobot_smolvla(
                checkpoint,
                vlm_base_path=vlm_base_path,
                stats_variant=stats_variant,
                load_device=options.pop("load_device", "cpu"),
                cache_dir=options.pop("cache_dir", None),
                revision=revision,
                local_files_only=options.pop("local_files_only", True),
                load_vlm_weights=options.pop("load_vlm_weights", True),
            )
        else:
            loader_options = {
                "vlm_base_path",
                "load_device",
                "cache_dir",
                "local_files_only",
                "load_vlm_weights",
            }.intersection(options)
            if loader_options:
                names = ", ".join(sorted(loader_options))
                raise TypeError(
                    f"injected SmolVLA policy does not accept loader option(s): {names}"
                )
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"unknown SmolVLA package option(s): {names}")

        from .package import build_smolvla_package

        components = build_smolvla_package(
            policy,
            checkpoint=checkpoint,
            revision=revision,
            stats_variant=stats_variant,
            default_num_steps=default_num_steps,
        )
        self._policy = components.policy
        self._module = components.module
        self._processor = components.processor
        self._spec = components.spec
        return components.package

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        return self.processor.preprocess_one(request)

    def preprocess_lerobot_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.processor.from_lerobot_batch(batch)

    def synthetic_batch(
        self,
        *,
        batch_size: int = 1,
        language_length: int = 48,
        seed: int = 0,
    ) -> Mapping[str, Any]:
        return self.processor.synthetic_batch(
            batch_size=batch_size,
            language_length=language_length,
            seed=seed,
        )


__all__ = [
    "DEFAULT_SMOLVLA_REVISION",
    "SmolVLAAdapter",
    "load_lerobot_smolvla",
]
