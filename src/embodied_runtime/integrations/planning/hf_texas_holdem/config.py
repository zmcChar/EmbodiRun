"""Validated runtime settings for the Hugging Face planner."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


@dataclass(frozen=True, slots=True)
class HfTexasHoldemPlannerConfig:
    """Placement, loading, and generation settings for one planner model."""

    checkpoint: str
    device: str = "cuda"
    dtype: str = "float16"
    model_kind: str = "image_text_to_text"
    max_new_tokens: int = 512
    plan_ttl_s: float = 300.0
    local_files_only: bool = True
    cache_dir: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = "sdpa"
    skill: str = "pick_and_place_poker"
    allow_single_json_fence: bool = False

    def __post_init__(self) -> None:
        for field_name in ("checkpoint", "device", "dtype", "model_kind", "skill"):
            raw = getattr(self, field_name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            object.__setattr__(self, field_name, raw.strip())
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("dtype must be one of: auto, float16, bfloat16, float32")
        if self.model_kind not in {"image_text_to_text", "causal_lm"}:
            raise ValueError("model_kind must be 'image_text_to_text' or 'causal_lm'")
        if not _IDENTIFIER.fullmatch(self.skill):
            raise ValueError("skill must be a valid planning identifier")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if isinstance(self.plan_ttl_s, bool) or not isinstance(self.plan_ttl_s, (int, float)):
            raise TypeError("plan_ttl_s must be a real number")
        normalized_ttl = float(self.plan_ttl_s)
        if not math.isfinite(normalized_ttl) or normalized_ttl <= 0.0:
            raise ValueError("plan_ttl_s must be finite and greater than zero")
        object.__setattr__(self, "plan_ttl_s", normalized_ttl)
        for field_name in (
            "local_files_only",
            "trust_remote_code",
            "allow_single_json_fence",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool")
        for field_name in ("cache_dir", "revision", "attn_implementation"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} must be None or a non-empty string")
            if isinstance(value, str):
                object.__setattr__(self, field_name, value.strip())


__all__ = ["HfTexasHoldemPlannerConfig"]
