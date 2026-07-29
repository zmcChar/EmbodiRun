"""Formal adapter for NVIDIA GR00T N1.7.

The official :class:`gr00t.policy.gr00t_policy.Gr00tPolicy` owns GR00T's
processor, normalization, internal flow-matching loop, and action decoding.
This adapter deliberately treats that policy as one semantic forward stage.
The execution backend still owns placement of ``policy.model`` and the engine
still owns request batching and lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from embodied_runtime.contracts import (
    ActionChunk,
    EntrypointSpec,
    ModelPackage,
    ModelPackageError,
    ModelSpec,
    RawRequest,
    SingleForwardPlan,
    TensorTree,
)

from ...registry import register_model
from ..base import VLAAdapterBase

DEFAULT_CHECKPOINT = "nvidia/GR00T-N1.7-3B"
DEFAULT_EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
# Short spelling retained for callers that already imported the prototype.
DEFAULT_EMBODIMENT = DEFAULT_EMBODIMENT_TAG
DEFAULT_LANGUAGE_KEY = "annotation.language.language_instruction"
DEFAULT_ACTION_KEYS = ("eef_9d", "gripper_position", "joint_position")
DEFAULT_ACTION_DIM = 17
DEFAULT_ACTION_HORIZON = 40


def require_supported_embodiment(embodiment_tag: str) -> str:
    """Validate the one embodiment covered by this first integration slice."""

    normalized = str(embodiment_tag).strip()
    if normalized != DEFAULT_EMBODIMENT_TAG:
        raise ValueError(
            "the initial GR00T N1.7 integration supports only "
            f"{DEFAULT_EMBODIMENT_TAG!r}; got {embodiment_tag!r}"
        )
    return normalized


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - GR00T itself requires torch
        raise ModelPackageError("GR00T N1.7 requires PyTorch and NVIDIA Isaac-GR00T") from error
    return torch


def _require_gr00t_policy():
    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except ImportError as error:
        raise ModelPackageError(
            "GR00T N1.7 requires NVIDIA Isaac-GR00T; install the official "
            "Isaac-GR00T package in this endpoint environment"
        ) from error
    return Gr00tPolicy


def _download_hf_snapshot(
    checkpoint: str,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ModelPackageError(
            "resolving a Hugging Face GR00T checkpoint requires huggingface-hub"
        ) from error
    try:
        return Path(
            snapshot_download(
                repo_id=checkpoint,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    except Exception as error:
        location = "local cache" if local_files_only else "Hugging Face Hub"
        raise ModelPackageError(
            f"could not resolve GR00T checkpoint {checkpoint!r} from {location}: {error}"
        ) from error


def _resolve_checkpoint(
    checkpoint: str | Path,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    """Resolve the outer GR00T snapshot, not its nested Cosmos dependency."""

    path = Path(checkpoint).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists():
        raise ModelPackageError(f"local GR00T checkpoint must be a directory, got {path}")
    return _download_hf_snapshot(
        str(checkpoint),
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )


def load_gr00t_n17(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    *,
    embodiment_tag: str = DEFAULT_EMBODIMENT_TAG,
    load_device: str = "cpu",
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = True,
    strict: bool = True,
):
    """Load NVIDIA's official policy from a local or Hugging Face checkpoint.

    The normal formal path loads on CPU.  The selected hardware backend later
    moves ``policy.model`` to its concrete device; accepting a non-CPU load
    device here would silently duplicate that responsibility. ``local_files_only``
    applies to the outer GR00T snapshot. NVIDIA's current policy resolves its
    nested Cosmos backbone independently; the verified Transformers stack may
    still query Hub metadata while constructing a cached tokenizer, so this
    flag alone does not guarantee a fully air-gapped launch.
    """

    embodiment_tag = require_supported_embodiment(embodiment_tag)
    if str(load_device).strip().lower() != "cpu":
        raise ModelPackageError(
            "GR00T must load on CPU so the execution backend owns device placement"
        )
    source = _resolve_checkpoint(
        checkpoint,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    policy_type = _require_gr00t_policy()
    try:
        return policy_type(
            embodiment_tag=embodiment_tag,
            model_path=str(source),
            device="cpu",
            strict=strict,
        )
    except Exception as error:
        raise ModelPackageError(
            f"failed to load GR00T N1.7 policy from {source}: {error}"
        ) from error


def synthetic_droid_request(
    prompt: str = "pick up the object",
    *,
    image_height: int = 180,
    image_width: int = 320,
) -> RawRequest:
    """Create one correctly shaped DROID request for smoke tests.

    The zero-valued observation validates plumbing, loading, and inference; it
    is not a meaningful robotics benchmark sample.
    """

    if not prompt:
        raise ValueError("prompt must not be empty")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image_height and image_width must be greater than zero")
    frame_history = np.zeros(
        (2, image_height, image_width, 3),
        dtype=np.uint8,
    )
    eef_9d = np.zeros((1, 9), dtype=np.float32)
    # A valid identity 6D rotation follows the XYZ translation.  Keeping this
    # state physically valid matters because relative-action decoding may
    # convert it back to a rotation matrix.
    eef_9d[..., 3:] = np.asarray((1, 0, 0, 0, 1, 0), dtype=np.float32)
    return RawRequest(
        observation={
            "video": {
                "exterior_image_1_left": frame_history.copy(),
                "wrist_image_left": frame_history.copy(),
            },
            "state": {
                "eef_9d": eef_9d,
                "gripper_position": np.zeros((1, 1), dtype=np.float32),
                "joint_position": np.zeros((1, 7), dtype=np.float32),
            },
        },
        prompt=prompt,
    )


def _config_values(config: Any, attribute: str) -> tuple[Any, ...]:
    if config is None:
        return ()
    value = getattr(config, attribute, None)
    if value is None and isinstance(config, Mapping):
        value = config.get(attribute)
    if value is None:
        return ()
    return tuple(value)


def _policy_modality_config(policy: Any, modality: str) -> Any | None:
    configs = getattr(policy, "modality_configs", None)
    if not isinstance(configs, Mapping):
        return None
    return configs.get(modality)


def _embodiment_value(value: Any) -> str:
    resolved = getattr(value, "value", value)
    return str(resolved)


def _as_single_video(value: Any, *, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ModelPackageError(f"{path} must be a numpy.ndarray")
    if value.dtype != np.uint8:
        raise ModelPackageError(f"{path} must have dtype uint8, got {value.dtype}")
    if value.ndim == 4:
        value = value[None, ...]
    elif value.ndim != 5 or value.shape[0] != 1:
        raise ModelPackageError(
            f"{path} must have shape [T,H,W,C] or [1,T,H,W,C], got {value.shape}"
        )
    if value.shape[-1] != 3:
        raise ModelPackageError(f"{path} must contain RGB images, got shape {value.shape}")
    return np.ascontiguousarray(value)


def _as_single_state(value: Any, *, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ModelPackageError(f"{path} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise ModelPackageError(f"{path} must have dtype float32, got {value.dtype}")
    if value.ndim == 2:
        value = value[None, ...]
    elif value.ndim != 3 or value.shape[0] != 1:
        raise ModelPackageError(f"{path} must have shape [T,D] or [1,T,D], got {value.shape}")
    return np.ascontiguousarray(value)


def _as_single_language(value: Any, *, path: str) -> list[list[str]]:
    if isinstance(value, str):
        return [[value]]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ModelPackageError(f"{path} must be a string, [string], or [[string]]")
    if len(value) == 1 and isinstance(value[0], str):
        return [[value[0]]]
    if (
        len(value) == 1
        and isinstance(value[0], (list, tuple))
        and len(value[0]) == 1
        and isinstance(value[0][0], str)
    ):
        return [[value[0][0]]]
    raise ModelPackageError(f"{path} must describe one request with one instruction")


def _same_keys(
    samples: Sequence[Mapping[str, Any]],
    *,
    path: str,
) -> tuple[str, ...]:
    keys = tuple(samples[0])
    expected = set(keys)
    for index, sample in enumerate(samples[1:], start=1):
        if set(sample) != expected:
            raise ModelPackageError(
                f"cannot collate {path}: sample {index} keys differ from sample 0"
            )
    return keys


def _collate_arrays(
    samples: Sequence[Mapping[str, np.ndarray]],
    *,
    path: str,
) -> dict[str, np.ndarray]:
    keys = _same_keys(samples, path=path)
    result: dict[str, np.ndarray] = {}
    for key in keys:
        values = tuple(sample[key] for sample in samples)
        if any(not isinstance(value, np.ndarray) for value in values):
            raise ModelPackageError(f"cannot collate {path}.{key}: expected numpy arrays")
        if any(value.ndim == 0 or value.shape[0] != 1 for value in values):
            shapes = tuple(value.shape for value in values)
            raise ModelPackageError(
                f"cannot collate {path}.{key}: samples must retain B=1, got {shapes}"
            )
        try:
            result[key] = np.concatenate(values, axis=0)
        except ValueError as error:
            raise ModelPackageError(f"cannot collate {path}.{key}: {error}") from error
    return result


class _Gr00tPolicyForward:
    """Callable semantic stage; the policy's model is placed by the backend."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        actions, info = self.policy.get_action(dict(observation))
        if not isinstance(actions, Mapping):
            raise ModelPackageError("GR00T policy get_action must return an action mapping")
        if not isinstance(info, Mapping):
            raise ModelPackageError("GR00T policy get_action info must be a mapping")
        return {"actions": dict(actions), "info": dict(info)}


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

        torch = _torch()
        model = getattr(policy, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("GR00T policy.model must be torch.nn.Module")
        model.eval()

        self._policy = policy
        self._embodiment = _embodiment_value(
            getattr(policy, "embodiment_tag", self._requested_embodiment)
        )
        language_key = getattr(policy, "language_key", self._language_key)
        if not isinstance(language_key, str) or not language_key:
            raise ModelPackageError("GR00T policy.language_key must be a non-empty string")
        self._language_key = language_key

        action_config = _policy_modality_config(policy, "action")
        action_keys = _config_values(action_config, "modality_keys")
        if action_keys:
            self._action_keys = tuple(str(key) for key in action_keys)
        action_horizon = len(_config_values(action_config, "delta_indices"))
        if action_horizon == 0:
            action_horizon = DEFAULT_ACTION_HORIZON
        action_dim = (
            int(requested_action_dim) if requested_action_dim is not None else DEFAULT_ACTION_DIM
        )
        if action_dim <= 0:
            raise ModelPackageError("GR00T action_dim must be greater than zero")

        model_id = str(checkpoint) if checkpoint else "gr00t-n1.7-injected-policy"
        self._spec = ModelSpec(
            model_id=model_id,
            family="gr00t_n17",
            revision=revision,
            modalities=("vision", "language", "proprioception"),
            action_dim=action_dim,
            action_horizon=action_horizon,
            metadata={
                "framework": "torch",
                "reference_loader": "NVIDIA Isaac-GR00T",
                "execution_plan": "single_forward",
                "embodiment_tag": self._embodiment,
                "language_key": self._language_key,
                "action_keys": self._action_keys,
                "internal_action_dim": int(
                    getattr(getattr(model, "config", None), "max_action_dim", 132)
                ),
            },
        )
        plan = SingleForwardPlan()
        forward = _Gr00tPolicyForward(policy)
        return ModelPackage(
            spec=self._spec,
            checkpoint=checkpoint or None,
            plan=plan,
            entrypoints={plan.forward: forward},
            entrypoint_specs={
                plan.forward: EntrypointSpec(
                    plan.forward,
                    "Run NVIDIA's official GR00T preprocessing, flow inference, and decoding.",
                    batchable=True,
                    safe_point_after=True,
                    metadata={
                        "output": "named_action_chunk",
                        "internal_generation": "flow_matching",
                    },
                )
            },
            metadata={
                "framework": "torch",
                "runtime_module": model,
                "portable": False,
                "reference_runtime": "NVIDIA Isaac-GR00T",
                "embodiment_tag": self._embodiment,
                "language_key": self._language_key,
                "action_keys": self._action_keys,
            },
        )

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        observation = request.observation
        video = observation.get("video")
        state = observation.get("state")
        language = observation.get("language", {})
        if not isinstance(video, Mapping) or not video:
            raise ModelPackageError("GR00T observation['video'] must be a non-empty mapping")
        if not isinstance(state, Mapping) or not state:
            raise ModelPackageError("GR00T observation['state'] must be a non-empty mapping")
        if not isinstance(language, Mapping):
            raise ModelPackageError("GR00T observation['language'] must be a mapping")

        canonical_video = {
            str(key): _as_single_video(value, path=f"observation.video.{key}")
            for key, value in video.items()
        }
        canonical_state = {
            str(key): _as_single_state(value, path=f"observation.state.{key}")
            for key, value in state.items()
        }
        canonical_language = {
            str(key): _as_single_language(value, path=f"observation.language.{key}")
            for key, value in language.items()
        }
        if self._language_key not in canonical_language:
            if request.prompt is None:
                raise ModelPackageError(
                    f"GR00T requires observation['language'][{self._language_key!r}] "
                    "or RawRequest.prompt"
                )
            canonical_language[self._language_key] = [[request.prompt]]

        return {
            "video": canonical_video,
            "state": canonical_state,
            "language": canonical_language,
        }

    def collate(self, samples: Sequence[TensorTree]) -> Mapping[str, Any]:
        if not samples:
            raise ModelPackageError("collate requires at least one GR00T sample")
        if any(not isinstance(sample, Mapping) for sample in samples):
            raise ModelPackageError("GR00T samples must be nested mappings")
        typed_samples = tuple(samples)
        if any(set(sample) != {"video", "state", "language"} for sample in typed_samples):
            raise ModelPackageError("GR00T samples must contain exactly video, state, and language")

        video_samples = tuple(sample["video"] for sample in typed_samples)
        state_samples = tuple(sample["state"] for sample in typed_samples)
        language_samples = tuple(sample["language"] for sample in typed_samples)
        if any(not isinstance(value, Mapping) for value in video_samples):
            raise ModelPackageError("GR00T video samples must be mappings")
        if any(not isinstance(value, Mapping) for value in state_samples):
            raise ModelPackageError("GR00T state samples must be mappings")
        if any(not isinstance(value, Mapping) for value in language_samples):
            raise ModelPackageError("GR00T language samples must be mappings")

        language_keys = _same_keys(language_samples, path="language")
        collated_language: dict[str, list[list[str]]] = {}
        for key in language_keys:
            values = tuple(sample[key] for sample in language_samples)
            if any(
                not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list)
                for value in values
            ):
                raise ModelPackageError(f"cannot collate language.{key}: samples must retain B=1")
            collated_language[key] = [item for value in values for item in value]

        return {
            "video": _collate_arrays(video_samples, path="video"),
            "state": _collate_arrays(state_samples, path="state"),
            "language": collated_language,
        }

    def unbatch(self, outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not isinstance(outputs, Mapping):
            raise ModelPackageError("GR00T output must be a mapping")
        actions = outputs.get("actions")
        info = outputs.get("info", {})
        if not isinstance(actions, Mapping) or not actions:
            raise ModelPackageError("GR00T output['actions'] must be a non-empty mapping")
        if not isinstance(info, Mapping):
            raise ModelPackageError("GR00T output['info'] must be a mapping")

        samples: list[Mapping[str, Any]] = []
        for index in range(batch_size):
            sample_actions: dict[str, Any] = {}
            for key, value in actions.items():
                shape = getattr(value, "shape", None)
                if shape is None or len(shape) == 0 or int(shape[0]) != batch_size:
                    raise ModelPackageError(
                        f"cannot unbatch actions.{key}: expected leading dimension "
                        f"{batch_size}, got {shape}"
                    )
                sample_actions[str(key)] = value[index]
            samples.append({"actions": sample_actions, "info": dict(info)})
        return tuple(samples)

    def postprocess_one(self, output: TensorTree) -> ActionChunk:
        if not isinstance(output, Mapping):
            raise ModelPackageError("GR00T unbatched output must be a mapping")
        actions = output.get("actions")
        info = output.get("info", {})
        if not isinstance(actions, Mapping) or not actions:
            raise ModelPackageError("GR00T unbatched output['actions'] must be a non-empty mapping")
        normalized: dict[str, Any] = {}
        for key, value in actions.items():
            ndim = getattr(value, "ndim", None)
            if ndim != 2:
                raise ModelPackageError(
                    f"GR00T action {key!r} must have unbatched shape [T,D], "
                    f"got {getattr(value, 'shape', None)}"
                )
            normalized[str(key)] = value
        return ActionChunk(
            actions=normalized,
            metadata={
                "embodiment_tag": self._embodiment,
                "action_keys": tuple(normalized),
                "policy_info": dict(info) if isinstance(info, Mapping) else {},
            },
        )


__all__ = [
    "DEFAULT_ACTION_DIM",
    "DEFAULT_ACTION_HORIZON",
    "DEFAULT_ACTION_KEYS",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_EMBODIMENT",
    "DEFAULT_EMBODIMENT_TAG",
    "DEFAULT_LANGUAGE_KEY",
    "Gr00tN17Adapter",
    "load_gr00t_n17",
    "require_supported_embodiment",
    "synthetic_droid_request",
]
