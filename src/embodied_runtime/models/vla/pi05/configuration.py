"""Strict construction of LeRobot PI05Config from checkpoint JSON."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from ...errors import ModelPackageError
from .dependencies import require_lerobot


def manual_pi05_config(
    config_file: Path,
    *,
    load_device: str,
    load_dtype: str,
):
    """Parse LeRobot's config without draccus.

    LeRobot 0.5.1's draccus parser fails on Python 3.14 when resolving modern
    ``dict[...] | None`` annotations. Constructing the public PI05Config
    dataclass directly is equivalent.
    """

    (
        PI05Config,
        _,
        PolicyFeature,
        FeatureType,
        NormalizationMode,
        RTCConfig,
    ) = require_lerobot()
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
    except Exception as error:
        raise ModelPackageError(
            f"failed to construct PI05Config from {config_file}: {error}"
        ) from error
