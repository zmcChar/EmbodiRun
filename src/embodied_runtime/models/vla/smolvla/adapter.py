"""SmolVLA ModelAdapter with verified legacy LeRobot checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.metadata
from pathlib import Path
from typing import Any

import torch

from embodied_runtime.contracts import (
    EntrypointSpec,
    ModelPackage,
    ModelPackageError,
    ModelSpec,
    RawRequest,
)

from ...registry import register_model
from ..base import VLAAdapterBase
from .modeling_smolvla import SmolVLAFlowPlan, SmolVLAReferenceModule
from .processing_smolvla import SmolVLAProcessor

DEFAULT_SMOLVLA_REVISION = "3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c"
EXPECTED_LEROBOT_VERSION = "0.3.3"
_WEIGHTS_FILENAME = "model.safetensors"
_VERIFICATION_TENSORS = {
    "model._orig_mod.action_in_proj.weight": "model.action_in_proj.weight",
    (
        "model._orig_mod.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight"
    ): "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
    ("model._orig_mod.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"): (
        "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight"
    ),
}


def _require_lerobot_033() -> tuple[Any, Any]:
    try:
        installed = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise ModelPackageError(
            "SmolVLA requires the isolated 'smolvla' extra with lerobot==0.3.3"
        ) from error
    if installed != EXPECTED_LEROBOT_VERSION:
        raise ModelPackageError(
            "the legacy SmolVLA adapter requires lerobot=="
            f"{EXPECTED_LEROBOT_VERSION}, found {installed}; keep it in an "
            "endpoint environment separate from the pi0.5 runtime"
        )
    try:
        # Importing the policy registers its config subclass before generic
        # PreTrainedConfig resolution.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as error:
        raise ModelPackageError(
            "SmolVLA requires lerobot[smolvla]==0.3.3 and transformers"
        ) from error
    return PreTrainedConfig, SmolVLAPolicy


def _checkpoint_source(checkpoint: str | Path) -> tuple[str, Path | None]:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.name != _WEIGHTS_FILENAME:
            raise ModelPackageError(f"SmolVLA checkpoint file must be named {_WEIGHTS_FILENAME!r}")
        return str(path.parent), path
    if path.is_dir():
        weights = path / _WEIGHTS_FILENAME
        if not weights.is_file():
            raise ModelPackageError(f"local SmolVLA checkpoint directory has no {weights.name}")
        if not (path / "config.json").is_file():
            raise ModelPackageError("local SmolVLA checkpoint directory has no config.json")
        return str(path), weights
    return str(checkpoint), None


def _resolve_remote_weights(
    checkpoint: str,
    *,
    cache_dir: str | None,
    revision: str,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ModelPackageError(
            "resolving a Hugging Face SmolVLA checkpoint requires huggingface-hub"
        ) from error
    try:
        return Path(
            hf_hub_download(
                repo_id=checkpoint,
                filename=_WEIGHTS_FILENAME,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    except Exception as error:
        location = "local cache" if local_files_only else "Hugging Face Hub"
        raise ModelPackageError(
            f"could not resolve SmolVLA weights for {checkpoint!r} from {location}: {error}"
        ) from error


def _normalization_tensor_keys(stats_variant: str) -> dict[str, dict[str, str]]:
    variant = stats_variant.strip()
    if not variant:
        raise ValueError("stats_variant must not be empty")
    return {
        "observation.state": {
            "mean": f"normalize_inputs.{variant}_buffer_observation_state.mean",
            "std": f"normalize_inputs.{variant}_buffer_observation_state.std",
        },
        "action": {
            "mean": f"unnormalize_outputs.{variant}_buffer_action.mean",
            "std": f"unnormalize_outputs.{variant}_buffer_action.std",
        },
    }


def _load_dataset_stats(
    weights_file: Path,
    stats_variant: str,
) -> dict[str, dict[str, torch.Tensor]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError("SmolVLA checkpoint loading requires safetensors") from error

    selected = _normalization_tensor_keys(stats_variant)
    stats: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(weights_file), framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        required = {
            tensor_key
            for statistic_keys in selected.values()
            for tensor_key in statistic_keys.values()
        }
        missing = sorted(required - available)
        if missing:
            raise ModelPackageError(
                f"SmolVLA checkpoint lacks {stats_variant!r} statistics: {missing}"
            )
        for feature, statistic_keys in selected.items():
            stats[feature] = {
                statistic: checkpoint.get_tensor(tensor_key).float()
                for statistic, tensor_key in statistic_keys.items()
            }
    return stats


def _verify_loaded_weights(
    policy: torch.nn.Module,
    weights_file: Path,
    stats_variant: str,
) -> None:
    """Reject partial/random loads hidden by LeRobot's legacy strict=False path."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError("SmolVLA checkpoint verification requires safetensors") from error

    verification = dict(_VERIFICATION_TENSORS)
    verification.update(
        {
            f"normalize_inputs.{stats_variant}_buffer_observation_state.mean": (
                "normalize_inputs.buffer_observation_state.mean"
            ),
            f"unnormalize_outputs.{stats_variant}_buffer_action.std": (
                "unnormalize_outputs.buffer_action.std"
            ),
        }
    )
    actual_state = policy.state_dict()
    with safe_open(str(weights_file), framework="pt", device="cpu") as checkpoint:
        available = set(checkpoint.keys())
        for source_key, destination_key in verification.items():
            if source_key not in available:
                raise ModelPackageError(
                    f"SmolVLA checkpoint has no verification tensor {source_key!r}"
                )
            actual = actual_state.get(destination_key)
            if actual is None:
                raise ModelPackageError(f"loaded SmolVLA policy has no tensor {destination_key!r}")
            expected = checkpoint.get_tensor(source_key)
            actual_cpu = actual.detach().to(device="cpu")
            if actual_cpu.shape != expected.shape or not torch.equal(
                actual_cpu,
                expected.to(actual_cpu.dtype),
            ):
                raise ModelPackageError(
                    "SmolVLA checkpoint sentinel did not match after loading: "
                    f"{source_key!r} -> {destination_key!r}"
                )


def _validate_vlm_source(vlm_base_path: str | Path, *, load_weights: bool) -> str:
    path = Path(vlm_base_path).expanduser()
    if not path.is_dir():
        raise ModelPackageError(f"local SmolVLM2 base directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise ModelPackageError(f"local SmolVLM2 base has no config.json: {path}")
    if load_weights and not (path / _WEIGHTS_FILENAME).is_file():
        raise ModelPackageError(f"local SmolVLM2 base has no {_WEIGHTS_FILENAME}: {path}")
    return str(path.resolve())


def load_lerobot_smolvla(
    checkpoint: str | Path,
    *,
    vlm_base_path: str | Path,
    stats_variant: str = "so100",
    load_device: str = "cpu",
    cache_dir: str | None = None,
    revision: str = DEFAULT_SMOLVLA_REVISION,
    local_files_only: bool = True,
    load_vlm_weights: bool = True,
) -> torch.nn.Module:
    """Load the verified pre-pipeline SmolVLA checkpoint with LeRobot 0.3.3."""

    PreTrainedConfig, SmolVLAPolicy = _require_lerobot_033()
    source, local_weights = _checkpoint_source(checkpoint)
    weights_file = local_weights or _resolve_remote_weights(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    dataset_stats = _load_dataset_stats(weights_file, stats_variant)
    vlm_source = _validate_vlm_source(
        vlm_base_path,
        load_weights=load_vlm_weights,
    )
    try:
        config = PreTrainedConfig.from_pretrained(
            source,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        config.device = load_device
        config.vlm_model_name = vlm_source
        config.load_vlm_weights = load_vlm_weights
        config.compile_model = False
        policy = SmolVLAPolicy.from_pretrained(
            source,
            config=config,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            dataset_stats=dataset_stats,
            # LeRobot 0.3.3 first attempts the literal _orig_mod keys, then its
            # SmolVLA override remaps and validates them. strict=True aborts
            # before that second stage.
            strict=False,
        ).eval()
    except Exception as error:
        raise ModelPackageError(
            f"failed to load SmolVLA weights from {source!r}: {error}"
        ) from error
    _verify_loaded_weights(policy, weights_file, stats_variant)
    return policy


@register_model("smolvla")
class SmolVLAAdapter(VLAAdapterBase):
    """Build a staged flow package around a verified SmolVLA policy."""

    def __init__(self) -> None:
        self._policy: torch.nn.Module | None = None
        self._processor: SmolVLAProcessor | None = None
        self._module: SmolVLAReferenceModule | None = None
        self._spec: ModelSpec | None = None

    @property
    def loaded_policy(self) -> torch.nn.Module:
        if self._policy is None:
            raise ModelPackageError("build_package must be called before accessing the policy")
        return self._policy

    @property
    def processor(self) -> SmolVLAProcessor:
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
        if not isinstance(policy, torch.nn.Module):
            raise TypeError("SmolVLA policy must be a torch.nn.Module")

        config = policy.config
        rtc_config = getattr(config, "rtc_config", None)
        if rtc_config is not None and bool(getattr(rtc_config, "enabled", False)):
            raise ModelPackageError(
                "RTC-enabled SmolVLA checkpoints are not supported by the staged adapter"
            )

        module = SmolVLAReferenceModule(policy)
        processor = SmolVLAProcessor(policy)
        action_dim = module.native_action_dim
        action_horizon = int(config.chunk_size)
        self._policy = policy
        self._module = module
        self._processor = processor
        self._spec = ModelSpec(
            model_id=str(checkpoint) if checkpoint else "smolvla-injected-policy",
            family="smolvla_flow",
            revision=revision,
            modalities=("vision", "language", "proprioception"),
            action_dim=action_dim,
            action_horizon=action_horizon,
            metadata={
                "framework": "torch",
                "reference_loader": f"lerobot=={EXPECTED_LEROBOT_VERSION}",
                "image_features": tuple(config.image_features),
                "internal_action_dim": int(config.max_action_dim),
                "stats_variant": stats_variant,
            },
        )
        plan = SmolVLAFlowPlan(
            default_num_steps=int(config.num_steps),
        )
        return ModelPackage(
            spec=self._spec,
            checkpoint=checkpoint or None,
            plan=plan,
            entrypoints={
                plan.encode: module.encode_prefix,
                plan.initialize: module.init_state,
                plan.step: module.denoise_step,
                plan.finalize: module.finalize,
            },
            entrypoint_specs={
                plan.encode: EntrypointSpec(
                    plan.encode,
                    "Normalize observations and encode SmolVLA prefix KV.",
                    batchable=True,
                    safe_point_after=True,
                ),
                plan.initialize: EntrypointSpec(
                    plan.initialize,
                    "Create the FP32 [B, horizon, 32] flow state.",
                    batchable=True,
                    safe_point_after=True,
                ),
                plan.step: EntrypointSpec(
                    plan.step,
                    "Return one SmolVLA action velocity field.",
                    batchable=True,
                    safe_point_after=True,
                    metadata={"output": "velocity", "state_update": "engine_owned_euler"},
                ),
                plan.finalize: EntrypointSpec(
                    plan.finalize,
                    "Crop action padding and apply checkpoint unnormalization.",
                    batchable=True,
                    safe_point_after=True,
                ),
            },
            metadata={
                "framework": "torch",
                "runtime_module": module,
                "step_output": "velocity",
                "state_update": "state + dt * velocity",
                "portable": False,
                "source": {
                    "lerobot_version": EXPECTED_LEROBOT_VERSION,
                    "lerobot_license": "Apache-2.0",
                    "smolvla_checkpoint_revision": revision,
                },
                "action_contract": {
                    "native_dim": action_dim,
                    "internal_padded_dim": int(config.max_action_dim),
                    "numeric_fusion_with_pi05": False,
                },
            },
        )

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        return self.processor.preprocess_one(request)

    def preprocess_lerobot_batch(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.processor.from_lerobot_batch(batch)

    def preprocess(
        self,
        requests: Sequence[RawRequest] | Mapping[str, Any],
    ) -> Mapping[str, Any]:
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
