"""OpenVLA-OFT snapshot resolution and exact component state routing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...errors import ModelPackageError

REQUIRED_METADATA_FILES = (
    "config.json",
    "dataset_statistics.json",
    "preprocessor_config.json",
)
WEIGHT_INDEX = "model.safetensors.index.json"
SINGLE_WEIGHTS = "model.safetensors"
COMPONENT_PREFIXES = {
    "vision_backbone.": "vision_backbone",
    "projector.": "projector",
    "language_model.": "language_model",
}


def read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelPackageError(f"could not read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ModelPackageError(f"{description} {path} must contain a JSON object")
    return value


def resolve_openvla_oft_checkpoint(
    checkpoint: str | Path,
    *,
    cache_dir: str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or one complete Hugging Face snapshot."""

    source = str(checkpoint)
    local = Path(source).expanduser()
    if local.is_dir():
        root = local.resolve()
    elif local.is_absolute() or source.startswith(("./", "../", "~")):
        raise ModelPackageError(f"OpenVLA-OFT checkpoint directory does not exist: {local}")
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise ModelPackageError(
                "resolving a Hugging Face OpenVLA-OFT checkpoint requires "
                "huggingface-hub; install the 'openvla_oft' extra"
            ) from error
        try:
            root = Path(
                snapshot_download(
                    repo_id=source,
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=(
                        "*.json",
                        "*.model",
                        "tokenizer.*",
                        "added_tokens.json",
                        "special_tokens_map.json",
                        "*.safetensors",
                    ),
                )
            )
        except Exception as error:
            mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub"
            raise ModelPackageError(
                f"could not resolve OpenVLA-OFT checkpoint {source!r} from {mode}: {error}"
            ) from error

    missing = [name for name in REQUIRED_METADATA_FILES if not (root / name).is_file()]
    if not (root / WEIGHT_INDEX).is_file() and not (root / SINGLE_WEIGHTS).is_file():
        missing.append(f"{WEIGHT_INDEX} or {SINGLE_WEIGHTS}")
    if missing:
        raise ModelPackageError(f"OpenVLA-OFT checkpoint {root} is missing: {', '.join(missing)}")
    return root


def load_checkpoint_components(
    root: Path,
    *,
    vision_backbone: Any,
    projector: Any,
    language_model: Any,
    load_device: str,
    load_dtype: Any,
    strict: bool,
) -> None:
    """Assign sharded safetensors into meta-initialized component modules."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint loading requires safetensors; install the 'openvla_oft' extra"
        ) from error

    components = {
        "vision_backbone": vision_backbone,
        "projector": projector,
        "language_model": language_model,
    }
    expected_parameters = {
        name: set(dict(module.named_parameters(remove_duplicate=False)))
        for name, module in components.items()
    }
    loaded: dict[str, set[str]] = {name: set() for name in components}
    ignored_prefixes = ("value_head.",)
    unknown_source_keys: list[str] = []

    for shard in _weight_shards(root):
        shard_state: dict[str, dict[str, Any]] = {name: {} for name in components}
        try:
            with safe_open(str(shard), framework="pt", device=load_device) as handle:
                for source_key in handle.keys():  # noqa: SIM118
                    target = _component_for_key(source_key)
                    if target is None:
                        if not source_key.startswith(ignored_prefixes):
                            unknown_source_keys.append(source_key)
                        continue
                    component_name, destination_key = target
                    tensor = handle.get_tensor(source_key)
                    if tensor.is_floating_point() and tensor.dtype != load_dtype:
                        tensor = tensor.to(dtype=load_dtype)
                    shard_state[component_name][destination_key] = tensor
        except Exception as error:
            raise ModelPackageError(
                f"could not read OpenVLA-OFT weight shard {shard}: {error}"
            ) from error

        for component_name, partial_state in shard_state.items():
            if not partial_state:
                continue
            try:
                incompatible = components[component_name].load_state_dict(
                    partial_state,
                    strict=False,
                    assign=True,
                )
            except Exception as error:
                raise ModelPackageError(
                    f"could not route {shard.name} into OpenVLA-OFT {component_name}: {error}"
                ) from error
            if incompatible.unexpected_keys:
                names = ", ".join(incompatible.unexpected_keys[:8])
                raise ModelPackageError(
                    f"OpenVLA-OFT {component_name} has incompatible checkpoint keys: {names}"
                )
            loaded[component_name].update(partial_state)

    missing_parameters = {
        name: sorted(parameters.difference(loaded[name]))
        for name, parameters in expected_parameters.items()
    }
    missing_parameters = {name: names for name, names in missing_parameters.items() if names}
    if missing_parameters:
        details = "; ".join(
            f"{name}: {', '.join(keys[:8])}" for name, keys in missing_parameters.items()
        )
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint left model parameters uninitialized: " + details
        )
    if strict and unknown_source_keys:
        names = ", ".join(sorted(unknown_source_keys)[:8])
        raise ModelPackageError(f"OpenVLA-OFT checkpoint has unknown component keys: {names}")

    tie_weights = getattr(language_model, "tie_weights", None)
    if callable(tie_weights):
        tie_weights()
    _reject_meta_values(components)


def _weight_shards(root: Path) -> tuple[Path, ...]:
    index_path = root / WEIGHT_INDEX
    if index_path.is_file():
        index = read_json_object(index_path, description="OpenVLA-OFT weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelPackageError(f"OpenVLA-OFT weight index {index_path} has no weight_map")
        shard_names = tuple(sorted({str(name) for name in weight_map.values()}))
        shards = tuple(root / name for name in shard_names)
    else:
        shards = (root / SINGLE_WEIGHTS,)
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise ModelPackageError(
            "OpenVLA-OFT checkpoint is missing weight shard(s): " + ", ".join(missing)
        )
    return shards


def _component_for_key(key: str) -> tuple[str, str] | None:
    for prefix, component in COMPONENT_PREFIXES.items():
        if key.startswith(prefix):
            return component, key[len(prefix) :]
    return None


def _reject_meta_values(components: Mapping[str, Any]) -> None:
    meta_parameters = [
        f"{component_name}.{parameter_name}"
        for component_name, module in components.items()
        for parameter_name, parameter in module.named_parameters()
        if parameter.is_meta
    ]
    if meta_parameters:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta parameters: " + ", ".join(meta_parameters[:8])
        )
    meta_buffers = [
        f"{component_name}.{buffer_name}"
        for component_name, module in components.items()
        for buffer_name, buffer in module.named_buffers()
        if buffer.is_meta
    ]
    if meta_buffers:
        raise ModelPackageError(
            "OpenVLA-OFT loader retained meta buffers: " + ", ".join(meta_buffers[:8])
        )
