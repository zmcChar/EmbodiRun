"""Validated model and action configuration for OpenVLA-OFT checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import ModelPackageError
from .checkpoint import read_json_object


@dataclass(frozen=True, slots=True)
class OpenVLAOFTModelConfig:
    text_config: dict[str, Any]
    use_fused_vision_backbone: bool
    num_images_in_input: int
    timm_model_ids: tuple[Any, ...]
    image_sizes: tuple[Any, ...]
    activation_layers: tuple[Any, ...]
    q01: list[Any]
    q99: list[Any]
    action_mask: list[Any]
    statistics_key: str
    action_dim: int
    action_horizon: int
    padded_vocab_size: int
    action_vocab_size: int
    n_action_bins: int


def load_model_config(
    root: Path,
    *,
    action_dim: int | None,
    action_horizon: int | None,
) -> OpenVLAOFTModelConfig:
    config = read_json_object(root / "config.json", description="OpenVLA-OFT config")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ModelPackageError("OpenVLA-OFT config.json has no text_config object")
    fused = bool(config.get("use_fused_vision_backbone", True))
    num_images_in_input = int(config.get("num_images_in_input") or 1)
    if num_images_in_input != 1:
        raise ModelPackageError(
            "the first OpenVLA-OFT adapter slice supports exactly one input image; "
            f"checkpoint config requests {num_images_in_input}"
        )
    try:
        timm_model_ids = tuple(config["timm_model_ids"])
        image_sizes = tuple(config.get("image_sizes", (224, 224)))
    except (KeyError, TypeError) as error:
        raise ModelPackageError("OpenVLA-OFT config lacks timm_model_ids/image_sizes") from error
    activation_layers = tuple(config.get("timm_override_act_layers", (None,) * len(timm_model_ids)))
    q01, q99, mask, statistics_key = action_statistics(
        root,
        config,
        action_dim=action_dim,
    )
    resolved_action_dim = len(q01)
    resolved_horizon = int(
        action_horizon if action_horizon is not None else config.get("num_action_chunks", 8)
    )
    if resolved_horizon <= 0:
        raise ModelPackageError("OpenVLA-OFT action_horizon must be greater than zero")
    pad_multiple = int(config.get("pad_to_multiple_of", 64))
    try:
        padded_vocab_size = int(text_config["vocab_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ModelPackageError("OpenVLA-OFT text_config has no valid vocab_size") from error
    action_vocab_size = padded_vocab_size - pad_multiple
    n_action_bins = int(config.get("n_action_bins", 256))
    if action_vocab_size <= 0:
        raise ModelPackageError("OpenVLA-OFT unpadded vocabulary size is invalid")

    return OpenVLAOFTModelConfig(
        text_config=text_config,
        use_fused_vision_backbone=fused,
        num_images_in_input=num_images_in_input,
        timm_model_ids=timm_model_ids,
        image_sizes=image_sizes,
        activation_layers=activation_layers,
        q01=q01,
        q99=q99,
        action_mask=mask,
        statistics_key=statistics_key,
        action_dim=resolved_action_dim,
        action_horizon=resolved_horizon,
        padded_vocab_size=padded_vocab_size,
        action_vocab_size=action_vocab_size,
        n_action_bins=n_action_bins,
    )


def action_statistics(
    root: Path,
    config: Mapping[str, Any],
    *,
    action_dim: int | None,
) -> tuple[list[Any], list[Any], list[Any], str]:
    dataset_statistics = read_json_object(
        root / "dataset_statistics.json",
        description="OpenVLA-OFT dataset statistics",
    )
    requested_key = config.get("unnorm_key")
    candidate_keys = (
        (
            str(requested_key),
            f"{requested_key}_no_noops",
        )
        if requested_key
        else ()
    )
    selected_key = next(
        (key for key in candidate_keys if key in dataset_statistics),
        next(iter(dataset_statistics), ""),
    )
    if not selected_key:
        raise ModelPackageError("OpenVLA-OFT dataset_statistics.json is empty")
    entry = dataset_statistics[selected_key]
    try:
        stats = entry["action"]
        q01 = list(stats["q01"])
        q99 = list(stats["q99"])
    except (KeyError, TypeError) as error:
        raise ModelPackageError(
            f"OpenVLA-OFT statistics {selected_key!r} have no action q01/q99"
        ) from error
    inferred_dim = len(q01)
    if len(q99) != inferred_dim:
        raise ModelPackageError("OpenVLA-OFT action q01/q99 lengths differ")
    if action_dim is not None and action_dim != inferred_dim:
        raise ModelPackageError(
            f"requested action_dim={action_dim}, but checkpoint statistics have {inferred_dim}"
        )
    mask = list(stats.get("mask", [True] * inferred_dim))
    if len(mask) != inferred_dim:
        raise ModelPackageError("OpenVLA-OFT action mask length differs from q01/q99")
    return q01, q99, mask, selected_key
