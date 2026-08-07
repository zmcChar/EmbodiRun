"""Resolution and verified loading of LeRobot pi0.5 checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...errors import ModelPackageError
from .configuration import manual_pi05_config
from .dependencies import require_lerobot

WEIGHTS_FILENAME = "model.safetensors"


def checkpoint_source(checkpoint: str | Path) -> tuple[str, Path | None]:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.name != WEIGHTS_FILENAME:
            raise ModelPackageError(f"pi0.5 checkpoint file must be named {WEIGHTS_FILENAME!r}")
        return str(path.parent), path
    if path.is_dir():
        weights = path / WEIGHTS_FILENAME
        if not weights.is_file():
            raise ModelPackageError(f"local pi0.5 checkpoint directory has no {weights.name}")
        return str(path), weights
    return str(checkpoint), None


def resolve_remote_weights(
    checkpoint: str,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> Path:
    try:
        from transformers.utils import cached_file
    except ImportError as error:
        raise ModelPackageError(
            "resolving a Hugging Face pi0.5 checkpoint requires transformers"
        ) from error
    resolved = cached_file(
        checkpoint,
        WEIGHTS_FILENAME,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved is None:
        raise ModelPackageError(f"could not resolve {WEIGHTS_FILENAME!r} for {checkpoint!r}")
    return Path(resolved)


def resolve_config_file(
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
    except ImportError as error:
        raise ModelPackageError(
            "resolving a Hugging Face pi0.5 config requires transformers"
        ) from error
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


def verify_loaded_weight(policy: Any, weights_file: Path) -> None:
    """Detect LeRobot's otherwise silent partial/random initialization."""

    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise ModelPackageError("pi0.5 checkpoint verification requires safetensors") from error
    source_keys = (
        (
            "paligemma_with_expert.paligemma.model.vision_tower."
            "vision_model.embeddings.patch_embedding.weight"
        ),
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
    """Load and verify real pi0.5 weights using LeRobot 0.5.1."""

    _, PI05Policy, *_ = require_lerobot()
    source, local_weights = checkpoint_source(checkpoint)
    config_file = resolve_config_file(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    config = manual_pi05_config(
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
    except Exception as error:
        raise ModelPackageError(f"failed to load pi0.5 weights from {source!r}: {error}") from error
    weights_file = local_weights or resolve_remote_weights(
        source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    verify_loaded_weight(policy, weights_file)
    return policy
