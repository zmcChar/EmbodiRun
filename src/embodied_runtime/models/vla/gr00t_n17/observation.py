"""GR00T DROID request normalization, batching, and action decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from embodied_runtime.types import TensorTree

from ...action import ActionChunk
from ...errors import ModelPackageError
from ...request import RawRequest


def preprocess_request(request: RawRequest, *, language_key: str) -> Mapping[str, Any]:
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
    if language_key not in canonical_language:
        if request.prompt is None:
            raise ModelPackageError(
                f"GR00T requires observation['language'][{language_key!r}] or RawRequest.prompt"
            )
        canonical_language[language_key] = [[request.prompt]]

    return {
        "video": canonical_video,
        "state": canonical_state,
        "language": canonical_language,
    }


def collate_samples(samples: Sequence[TensorTree]) -> Mapping[str, Any]:
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


def unbatch_outputs(outputs: TensorTree, batch_size: int) -> Sequence[TensorTree]:
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


def postprocess_output(output: TensorTree, *, embodiment: str) -> ActionChunk:
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
            "embodiment_tag": embodiment,
            "action_keys": tuple(normalized),
            "policy_info": dict(info) if isinstance(info, Mapping) else {},
        },
    )


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


def _same_keys(samples: Sequence[Mapping[str, Any]], *, path: str) -> tuple[str, ...]:
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
