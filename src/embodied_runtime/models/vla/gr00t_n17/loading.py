"""Lazy dependency resolution and official GR00T policy loading."""

from __future__ import annotations

from pathlib import Path

from ...errors import ModelPackageError
from .constants import DEFAULT_CHECKPOINT, DEFAULT_EMBODIMENT_TAG


def require_supported_embodiment(embodiment_tag: str) -> str:
    """Validate the one embodiment covered by this first integration slice."""

    normalized = str(embodiment_tag).strip()
    if normalized != DEFAULT_EMBODIMENT_TAG:
        raise ValueError(
            "the initial GR00T N1.7 integration supports only "
            f"{DEFAULT_EMBODIMENT_TAG!r}; got {embodiment_tag!r}"
        )
    return normalized


def require_gr00t_policy():
    try:
        from gr00t.policy.gr00t_policy import Gr00tPolicy
    except ImportError as error:
        raise ModelPackageError(
            "GR00T N1.7 requires NVIDIA Isaac-GR00T; install the official "
            "Isaac-GR00T package in this endpoint environment"
        ) from error
    return Gr00tPolicy


def download_hf_snapshot(
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


def resolve_checkpoint(
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
    return download_hf_snapshot(
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
    """Load NVIDIA's official policy from a local or Hugging Face checkpoint."""

    embodiment_tag = require_supported_embodiment(embodiment_tag)
    if str(load_device).strip().lower() != "cpu":
        raise ModelPackageError(
            "GR00T must load on CPU so the execution backend owns device placement"
        )
    source = resolve_checkpoint(
        checkpoint,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
    )
    policy_type = require_gr00t_policy()
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
