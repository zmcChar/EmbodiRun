"""Lightweight pi0.5 adapter orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...registry import register_model
from ...request import RawRequest
from ...spec import ModelSpec
from ..base import VLAAdapterBase
from .checkpoint import load_lerobot_pi05


@register_model("pi05")
class Pi05Adapter(VLAAdapterBase):
    """Build a portable staged-flow package around a real pi0.5 checkpoint."""

    def __init__(self) -> None:
        self._policy: Any | None = None
        self._processor: Any | None = None
        self._module: Any | None = None
        self._spec: ModelSpec | None = None

    @property
    def loaded_policy(self) -> Any:
        """The verified LeRobot policy, exposed for parity tests only."""

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
            model_id="pi05",
            family="pi05_flow",
            modalities=("vision", "language", "proprioception"),
            action_dim=32,
            action_horizon=50,
            metadata={
                "framework": "torch",
                "reference_loader": "lerobot==0.5.1",
            },
        )

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage:
        policy = options.pop("policy", None)
        if policy is None:
            policy = load_lerobot_pi05(
                checkpoint,
                load_device=options.pop("load_device", "cpu"),
                load_dtype=options.pop("load_dtype", "bfloat16"),
                cache_dir=options.pop("cache_dir", None),
                revision=options.get("revision"),
                local_files_only=options.pop("local_files_only", False),
                strict=options.pop("strict", True),
            )
        if options.keys() - {"revision"}:
            unknown = ", ".join(sorted(options.keys() - {"revision"}))
            raise TypeError(f"unknown pi0.5 package option(s): {unknown}")
        revision = options.get("revision")

        from .package import build_pi05_package

        components = build_pi05_package(
            policy,
            checkpoint=checkpoint,
            revision=revision,
        )
        self._policy = components.policy
        self._processor = components.processor
        self._module = components.module
        self._spec = components.spec
        return components.package

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        """Prepare one logical request while retaining tensor ``B=1``."""

        return self.processor.from_requests((request,))

    def preprocess_lerobot_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Explicit helper for callers that already own a collated LeRobot batch."""

        return self.processor.from_lerobot_batch(batch)

    def synthetic_batch(
        self,
        *,
        batch_size: int = 1,
        language_length: int = 48,
        seed: int = 0,
    ) -> Mapping[str, Any]:
        """Public real-weight smoke input requiring no tokenizer download."""

        return self.processor.synthetic_batch(
            batch_size=batch_size,
            language_length=language_length,
            seed=seed,
        )


__all__ = ["Pi05Adapter", "load_lerobot_pi05"]
