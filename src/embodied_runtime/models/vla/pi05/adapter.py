"""pi0.5 ModelAdapter and verified LeRobot checkpoint loading.

The adapter owns model identity, preprocessing, reference semantics, and
correctness metadata.  Scheduling, Euler integration, device placement,
compilation, streams, graph capture, and memory policy remain outside this
module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import torch

from embodied_runtime.contracts import (
    EntrypointSpec,
    IterativeFlowPlan,
    ModelPackage,
    ModelPackageError,
    ModelSpec,
    RawRequest,
)

from ...registry import register_model
from ..base import VLAAdapterBase
from .modeling_pi05 import Pi05ReferenceModule
from .processing_pi05 import Pi05Processor

_VVLA_SOURCE_COMMIT = "80b5cf48c8710c69ed97200903562e9787efe105"
_WEIGHTS_FILENAME = "model.safetensors"


def _require_lerobot():
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
    except ImportError as exc:
        raise ModelPackageError(
            "pi0.5 requires lerobot==0.5.1 plus its transformers dependency; "
            "install the project's 'pi05' extra"
        ) from exc
    return (
        PI05Config,
        PI05Policy,
        PolicyFeature,
        FeatureType,
        NormalizationMode,
        RTCConfig,
    )


def _checkpoint_source(checkpoint: str | Path) -> tuple[str, Path | None]:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.name != _WEIGHTS_FILENAME:
            raise ModelPackageError(f"pi0.5 checkpoint file must be named {_WEIGHTS_FILENAME!r}")
        return str(path.parent), path
    if path.is_dir():
        weights = path / _WEIGHTS_FILENAME
        if not weights.is_file():
            raise ModelPackageError(f"local pi0.5 checkpoint directory has no {weights.name}")
        return str(path), weights
    return str(checkpoint), None


def _resolve_remote_weights(
    checkpoint: str,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    try:
        from transformers.utils import cached_file
    except ImportError as exc:
        raise ModelPackageError(
            "resolving a Hugging Face pi0.5 checkpoint requires transformers"
        ) from exc
    resolved = cached_file(
        checkpoint,
        _WEIGHTS_FILENAME,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved is None:
        raise ModelPackageError(f"could not resolve {_WEIGHTS_FILENAME!r} for {checkpoint!r}")
    return Path(resolved)


def _resolve_config_file(
    source: str,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    local = Path(source)
    if local.is_dir():
        config_file = local / "config.json"
        if not config_file.is_file():
            raise ModelPackageError(f"local pi0.5 checkpoint has no {config_file.name}")
        return config_file
    try:
        from transformers.utils import cached_file
    except ImportError as exc:
        raise ModelPackageError(
            "resolving a Hugging Face pi0.5 config requires transformers"
        ) from exc
    resolved = cached_file(
        source,
        "config.json",
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved is None:
        raise ModelPackageError(f"could not resolve 'config.json' for {source!r}")
    return Path(resolved)


def _manual_pi05_config(
    config_file: Path,
    *,
    load_device: str,
    load_dtype: str,
):
    """Parse LeRobot's config without draccus.

    LeRobot 0.5.1's draccus parser fails on Python 3.14 when resolving modern
    ``dict[...] | None`` annotations.  Constructing the public PI05Config
    dataclass directly is equivalent and keeps this prototype runnable in the
    WSL Python 3.14 environment.
    """

    (
        PI05Config,
        _,
        PolicyFeature,
        FeatureType,
        NormalizationMode,
        RTCConfig,
    ) = _require_lerobot()
    with config_file.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if raw.get("type") not in (None, "pi05"):
        raise ModelPackageError(f"checkpoint config declares type {raw.get('type')!r}, not 'pi05'")
    raw.pop("type", None)

    def convert_features(value: Any):
        if value is None:
            return None
        return {
            name: PolicyFeature(
                type=FeatureType(feature["type"]),
                shape=tuple(feature["shape"]),
            )
            for name, feature in value.items()
        }

    raw["input_features"] = convert_features(raw.get("input_features"))
    raw["output_features"] = convert_features(raw.get("output_features"))
    if raw.get("normalization_mapping") is not None:
        raw["normalization_mapping"] = {
            name: NormalizationMode(mode) for name, mode in raw["normalization_mapping"].items()
        }
    for tuple_field in ("image_resolution", "optimizer_betas"):
        if raw.get(tuple_field) is not None:
            raw[tuple_field] = tuple(raw[tuple_field])
    if raw.get("pretrained_path") is not None:
        raw["pretrained_path"] = Path(raw["pretrained_path"])
    if raw.get("rtc_config") is not None:
        raw["rtc_config"] = RTCConfig(**raw["rtc_config"])

    raw["device"] = load_device
    raw["dtype"] = load_dtype
    raw["compile_model"] = False
    raw["gradient_checkpointing"] = False
    known = {field.name for field in fields(PI05Config)}
    unknown = sorted(set(raw).difference(known))
    if unknown:
        names = ", ".join(unknown)
        raise ModelPackageError(
            "pi0.5 config contains fields unsupported by the pinned adapter: "
            f"{names}; refusing to guess whether they change inference semantics"
        )
    try:
        return PI05Config(**raw)
    except Exception as exc:
        raise ModelPackageError(
            f"failed to construct PI05Config from {config_file}: {exc}"
        ) from exc


def _verify_loaded_weight(policy: torch.nn.Module, weights_file: Path) -> None:
    """Detect LeRobot's otherwise silent partial/random initialization.

    Sentinels span the vision tower, language tower, action expert, time
    conditioning, and action projections. This is intentionally broader than
    checking one convenient leaf because LeRobot 0.5.1 catches state-dict
    loading errors internally.
    """

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ModelPackageError("pi0.5 checkpoint verification requires safetensors") from exc
    source_keys = (
        "paligemma_with_expert.paligemma.model.vision_tower."
        "vision_model.embeddings.patch_embedding.weight",
        "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
        "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight",
        "time_mlp_in.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
    )
    actual_state = policy.state_dict()
    with safe_open(str(weights_file), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for source_key in source_keys:
            destination_key = f"model.{source_key}"
            if source_key not in available:
                raise ModelPackageError(
                    f"pi0.5 checkpoint has no verification tensor {source_key!r}"
                )
            actual = actual_state.get(destination_key)
            if actual is None:
                raise ModelPackageError(
                    f"loaded LeRobot policy has no parameter {destination_key!r}"
                )
            expected = handle.get_tensor(source_key)
            actual_cpu = actual.detach().to(device="cpu")
            expected_in_runtime_dtype = expected.to(actual_cpu.dtype)
            if actual_cpu.shape != expected.shape or not torch.equal(
                actual_cpu,
                expected_in_runtime_dtype,
            ):
                raise ModelPackageError(
                    "LeRobot returned a PI05Policy, but checkpoint sentinel "
                    f"{source_key!r} did not match; refusing a partial or random load"
                )


def load_lerobot_pi05(
    checkpoint: str | Path,
    *,
    load_device: str = "cpu",
    load_dtype: str = "bfloat16",
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    strict: bool = True,
):
    """Load and verify real pi0.5 weights using LeRobot 0.5.1.

    Device placement for inference still belongs to the hardware backend.
    ``load_device='cpu'`` is the default solely to avoid trusting a checkpoint's
    serialized development-device field (for example ``mps``).
    """

    _, PI05Policy, *_ = _require_lerobot()
    source, local_weights = _checkpoint_source(checkpoint)
    config_file = _resolve_config_file(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    config = _manual_pi05_config(
        config_file,
        load_device=load_device,
        load_dtype=load_dtype,
    )
    try:
        policy = PI05Policy.from_pretrained(
            source,
            config=config,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            strict=strict,
        ).eval()
    except Exception as exc:
        raise ModelPackageError(f"failed to load pi0.5 weights from {source!r}: {exc}") from exc
    weights_file = local_weights or _resolve_remote_weights(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    _verify_loaded_weight(policy, weights_file)
    return policy


@register_model("pi05")
class Pi05Adapter(VLAAdapterBase):
    """Build a portable staged-flow package around a real pi0.5 checkpoint."""

    def __init__(self) -> None:
        self._policy = None
        self._processor: Pi05Processor | None = None
        self._module: Pi05ReferenceModule | None = None
        self._spec: ModelSpec | None = None

    @property
    def loaded_policy(self):
        """The verified LeRobot policy, exposed for parity tests only."""

        return self._policy

    @property
    def processor(self) -> Pi05Processor:
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
        """Load weights and expose the four flow entrypoints.

        ``policy=...`` may inject an already-loaded LeRobot PI05Policy.  It is
        useful for parity tests because the reference and staged paths then share
        one copy of the 4.14B parameters.
        """

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
        config = policy.config
        rtc_config = getattr(config, "rtc_config", None)
        if rtc_config is not None and bool(getattr(rtc_config, "enabled", False)):
            raise ModelPackageError(
                "RTC-enabled pi0.5 checkpoints are not supported by the staged "
                "prototype because RTC changes the denoise-loop semantics"
            )
        self._policy = policy
        self._processor = Pi05Processor(policy)
        self._module = Pi05ReferenceModule(policy)
        output_features = getattr(config, "output_features", None) or {}
        action_feature = output_features.get("action")
        action_dim = (
            int(action_feature.shape[0])
            if action_feature is not None
            else int(config.max_action_dim)
        )
        model_id = str(checkpoint) if checkpoint else "pi05-injected-policy"
        self._spec = ModelSpec(
            model_id=model_id,
            family="pi05_flow",
            revision=revision,
            modalities=("vision", "language", "proprioception"),
            action_dim=action_dim,
            action_horizon=int(config.chunk_size),
            metadata={
                "framework": "torch",
                "reference_loader": "lerobot==0.5.1",
                "image_features": tuple(config.image_features),
                "image_resolution": tuple(config.image_resolution),
                "internal_action_dim": int(config.max_action_dim),
            },
        )
        plan = IterativeFlowPlan(default_num_steps=int(config.num_inference_steps))
        specs = {
            plan.encode: EntrypointSpec(
                plan.encode,
                "Encode images and tokens into per-layer prefix K/V.",
                batchable=True,
                safe_point_after=True,
            ),
            plan.initialize: EntrypointSpec(
                plan.initialize,
                "Create initial action noise; explicit noise overrides generator/seed.",
                batchable=True,
                safe_point_after=True,
            ),
            plan.step: EntrypointSpec(
                plan.step,
                "Return the pi0.5 velocity field for one flow time.",
                batchable=True,
                safe_point_after=True,
                metadata={"output": "velocity", "state_update": "engine_owned_euler"},
            ),
            plan.finalize: EntrypointSpec(
                plan.finalize,
                "Expose the integrated state as actions.",
                batchable=True,
                safe_point_after=True,
            ),
        }
        return ModelPackage(
            spec=self._spec,
            checkpoint=checkpoint or None,
            plan=plan,
            entrypoints={
                plan.encode: self._module.encode_prefix,
                plan.initialize: self._module.init_state,
                plan.step: self._module.denoise_step,
                plan.finalize: self._module.finalize,
            },
            entrypoint_specs=specs,
            metadata={
                "framework": "torch",
                "runtime_module": self._module,
                "step_output": "velocity",
                "state_update": "state + dt * velocity",
                "portable": False,
                "source": {
                    "vvla_commit": _VVLA_SOURCE_COMMIT,
                    "vvla_license": "MIT",
                    "lerobot_version": "0.5.1",
                    "lerobot_license": "Apache-2.0",
                    "openpi_origin": "Physical-Intelligence/openpi",
                },
            },
        )

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        """Prepare one logical request while retaining tensor ``B=1``."""

        return self.processor.from_requests((request,))

    def preprocess_lerobot_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        """Explicit helper for callers that already own a collated LeRobot batch."""

        return self.processor.from_lerobot_batch(batch)

    def preprocess(
        self,
        requests: Sequence[RawRequest] | Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Compatibility wrapper; new integrations use ``preprocess_one/collate``."""

        if isinstance(requests, Mapping):
            return self.preprocess_lerobot_batch(requests)
        return super().preprocess(requests)

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


Pi05ModelAdapter = Pi05Adapter


__all__ = ["Pi05Adapter", "Pi05ModelAdapter", "load_lerobot_pi05"]
