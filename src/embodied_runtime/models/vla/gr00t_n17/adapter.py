"""Lightweight orchestration for the formal GR00T N1.7 adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.types import TensorTree

from ...action import ActionChunk
from ...errors import ModelPackageError
from ...package import ModelPackage
from ...registry import register_model
from ...request import RawRequest
from ...spec import ModelSpec
from ..base import VLAAdapterBase
from .constants import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_ACTION_KEYS,
    DEFAULT_CHECKPOINT,
    DEFAULT_EMBODIMENT_TAG,
    DEFAULT_LANGUAGE_KEY,
)
from .loading import load_gr00t_n17, require_supported_embodiment


@register_model("gr00t_n17")
class Gr00tN17Adapter(VLAAdapterBase):
    """Expose GR00T N1.7 as a backend-managed single-forward model package."""

    def __init__(
        self,
        *,
        embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
        language_key: str = DEFAULT_LANGUAGE_KEY,
    ) -> None:
        embodiment_tag = require_supported_embodiment(embodiment_tag)
        if not language_key.strip():
            raise ValueError("language_key must not be empty")
        self._requested_embodiment = embodiment_tag
        self._embodiment = embodiment_tag
        self._language_key = language_key
        self._policy: Any | None = None
        self._spec: ModelSpec | None = None
        self._action_keys = DEFAULT_ACTION_KEYS

    @property
    def loaded_policy(self) -> Any:
        if self._policy is None:
            raise ModelPackageError("build_package must be called before accessing the policy")
        return self._policy

    @property
    def language_key(self) -> str:
        return self._language_key

    def describe(self) -> ModelSpec:
        if self._spec is not None:
            return self._spec
        return ModelSpec(
            model_id="gr00t-n1.7-3b",
            family="gr00t_n17",
            modalities=("vision", "language", "proprioception"),
            action_dim=DEFAULT_ACTION_DIM,
            action_horizon=DEFAULT_ACTION_HORIZON,
            metadata={
                "framework": "torch",
                "reference_loader": "NVIDIA Isaac-GR00T",
                "execution_plan": "single_forward",
                "embodiment_tag": self._requested_embodiment,
                "language_key": self._language_key,
                "action_keys": DEFAULT_ACTION_KEYS,
                "internal_action_dim": 132,
            },
        )

    def build_package(self, checkpoint: str = DEFAULT_CHECKPOINT, **options: Any) -> ModelPackage:
        policy = options.pop("policy", None)
        revision = options.pop("revision", None)
        requested_action_dim = options.pop("action_dim", None)
        if policy is None:
            policy = load_gr00t_n17(
                checkpoint,
                embodiment_tag=self._requested_embodiment,
                load_device=options.pop("load_device", "cpu"),
                cache_dir=options.pop("cache_dir", None),
                revision=revision,
                local_files_only=options.pop("local_files_only", True),
                strict=options.pop("strict", True),
            )
        else:
            loader_options = {
                "load_device",
                "cache_dir",
                "local_files_only",
                "strict",
            }.intersection(options)
            if loader_options:
                names = ", ".join(sorted(loader_options))
                raise TypeError(f"injected GR00T policy does not accept loader option(s): {names}")
        if options:
            names = ", ".join(sorted(options))
            raise TypeError(f"unknown GR00T N1.7 package option(s): {names}")

        from .package import build_gr00t_package

        components = build_gr00t_package(
            policy,
            checkpoint=checkpoint,
            revision=revision,
            requested_embodiment=self._requested_embodiment,
            requested_language_key=self._language_key,
            requested_action_dim=requested_action_dim,
        )
        self._policy = components.policy
        self._embodiment = components.embodiment
        self._language_key = components.language_key
        self._action_keys = components.action_keys
        self._spec = components.spec
        return components.package

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        from .observation import preprocess_request

        return preprocess_request(request, language_key=self._language_key)

    def collate(self, samples: Sequence[TensorTree]) -> Mapping[str, Any]:
        from .observation import collate_samples

        return collate_samples(samples)

    def unbatch(self, outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]:
        from .observation import unbatch_outputs

        return unbatch_outputs(outputs, batch_size)

    def postprocess_one(self, output: TensorTree) -> ActionChunk:
        from .observation import postprocess_output

        return postprocess_output(output, embodiment=self._embodiment)


__all__ = ["Gr00tN17Adapter", "load_gr00t_n17"]
