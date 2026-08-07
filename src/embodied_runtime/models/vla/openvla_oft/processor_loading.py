"""Tokenizer and image-normalization construction for OpenVLA-OFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import ModelPackageError


@dataclass(frozen=True, slots=True)
class OpenVLAOFTProcessorAssets:
    tokenizer: Any
    means: Any
    stds: Any
    image_size: int


def load_processor_assets(
    checkpoint: str | Path,
    *,
    cache_dir: str | None,
    revision: str | None,
    local_files_only: bool,
) -> OpenVLAOFTProcessorAssets:
    try:
        from transformers import AutoTokenizer
        from transformers.utils import cached_file
    except ImportError as error:
        raise ModelPackageError(
            "OpenVLA-OFT tokenization requires transformers; "
            "install the OpenVLA-OFT model dependencies"
        ) from error

    source = str(Path(checkpoint).expanduser()) if isinstance(checkpoint, Path) else checkpoint
    local = Path(source).expanduser()
    if local.is_dir():
        preprocessor_file = local / "preprocessor_config.json"
        if not preprocessor_file.is_file():
            raise ModelPackageError("local OpenVLA-OFT checkpoint has no preprocessor_config.json")
    elif local.is_absolute():
        raise ModelPackageError(f"local OpenVLA-OFT checkpoint directory does not exist: {local}")
    else:
        try:
            resolved = cached_file(
                source,
                "preprocessor_config.json",
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        except Exception as error:
            raise ModelPackageError(
                f"could not resolve OpenVLA-OFT preprocessor_config.json for {source!r}: {error}"
            ) from error
        if resolved is None:
            raise ModelPackageError(
                f"could not resolve OpenVLA-OFT preprocessor_config.json for {source!r}"
            )
        preprocessor_file = Path(resolved)

    try:
        with preprocessor_file.open(encoding="utf-8") as handle:
            config = json.load(handle)
        means = config["means"]
        stds = config["stds"]
        image_size = int(config["input_sizes"][0][-1])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ModelPackageError(
            f"invalid OpenVLA-OFT preprocessor config {preprocessor_file}: {error}"
        ) from error

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            padding_side="left",
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as error:
        raise ModelPackageError(
            f"could not load OpenVLA-OFT tokenizer from {source!r}: {error}"
        ) from error
    return OpenVLAOFTProcessorAssets(
        tokenizer=tokenizer,
        means=means,
        stds=stds,
        image_size=image_size,
    )
