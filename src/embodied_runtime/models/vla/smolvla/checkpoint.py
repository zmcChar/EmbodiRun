"""Resolution, normalization stats, and verified SmolVLA loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...errors import ModelPackageError
from .constants import (
    DEFAULT_SMOLVLA_REVISION,
    VERIFICATION_TENSORS,
    WEIGHTS_FILENAME,
)
from .dependencies import require_lerobot_033


def checkpoint_source(checkpoint: str | Path) -> tuple[str, Path | None]:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.name != WEIGHTS_FILENAME:
            raise ModelPackageError(f"SmolVLA checkpoint file must be named {WEIGHTS_FILENAME!r}")
        return str(path.parent), path
    if path.is_dir():
        weights = path / WEIGHTS_FILENAME
        if not weights.is_file():
            raise ModelPackageError(f"local SmolVLA checkpoint directory has no {weights.name}")
        if not (path / "config.json").is_file():
            raise ModelPackageError("local SmolVLA checkpoint directory has no config.json")
        return str(path), weights
    return str(checkpoint), None


def resolve_remote_weights(
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
                filename=WEIGHTS_FILENAME,
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


def normalization_tensor_keys(stats_variant: str) -> dict[str, dict[str, str]]:
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


def load_dataset_stats(weights_file: Path, stats_variant: str) -> dict[str, dict[str, Any]]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError("SmolVLA checkpoint loading requires safetensors") from error

    selected = normalization_tensor_keys(stats_variant)
    stats: dict[str, dict[str, Any]] = {}
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


def verify_loaded_weights(policy: Any, weights_file: Path, stats_variant: str) -> None:
    """Reject partial/random loads hidden by LeRobot's legacy strict=False path."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError("SmolVLA checkpoint verification requires safetensors") from error

    verification = dict(VERIFICATION_TENSORS)
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


def validate_vlm_source(vlm_base_path: str | Path, *, load_weights: bool) -> str:
    path = Path(vlm_base_path).expanduser()
    if not path.is_dir():
        raise ModelPackageError(f"local SmolVLM2 base directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise ModelPackageError(f"local SmolVLM2 base has no config.json: {path}")
    if load_weights and not (path / WEIGHTS_FILENAME).is_file():
        raise ModelPackageError(f"local SmolVLM2 base has no {WEIGHTS_FILENAME}: {path}")
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
) -> Any:
    """Load the verified pre-pipeline SmolVLA checkpoint with LeRobot 0.3.3."""

    import torch

    PreTrainedConfig, SmolVLAPolicy = require_lerobot_033()
    source, local_weights = checkpoint_source(checkpoint)
    weights_file = local_weights or resolve_remote_weights(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    dataset_stats = load_dataset_stats(weights_file, stats_variant)
    vlm_source = validate_vlm_source(vlm_base_path, load_weights=load_vlm_weights)
    try:
        config = PreTrainedConfig.from_pretrained(
            source,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            cli_overrides=[f"--device={load_device}"],
        )
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
            strict=False,
        ).eval()
    except Exception as error:
        raise ModelPackageError(
            f"failed to load SmolVLA weights from {source!r}: {error}"
        ) from error
    verify_loaded_weights(policy, weights_file, stats_variant)
    if torch.device(load_device).type == "cpu" and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    return policy
