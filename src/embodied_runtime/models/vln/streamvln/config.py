"""Validated configuration for the official StreamVLN checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .actions import MAX_FUTURE_ACTIONS

DEFAULT_MODEL = "mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
DEFAULT_SENSOR_CONFIG: dict[str, object] = {
    "rgb_height": 1.25,
    "camera_intrinsic": (
        (192.0, 0.0, 191.42857143, 0.0),
        (0.0, 192.0, 191.42857143, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
}


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cuda_fraction(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(  # noqa: TRY004
            "cuda_memory_fraction must be a number in (0, 1]"
        )
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("cuda_memory_fraction must be a finite number in (0, 1]")
    return result


def _checkpoint_name(value: str | Path) -> str:
    if isinstance(value, Path):
        text = str(value.expanduser())
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise TypeError("model_path must be a string or pathlib.Path")
    if not text:
        raise ValueError("model_path must not be empty")
    candidate = Path(text).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return text


@dataclass(frozen=True, slots=True)
class StreamVLNConfig:
    """Validated settings for one checkpoint-owning runtime process."""

    streamvln_root: str | Path
    model_path: str | Path = DEFAULT_MODEL
    device: str = "cuda:0"
    cuda_memory_fraction: float | None = None
    num_future_steps: int = MAX_FUTURE_ACTIONS
    num_frames: int = 32
    num_history: int = 8
    model_max_length: int = 4096
    max_new_tokens: int = 10_000
    dtype: str = "bfloat16"
    attn_implementation: str | None = "sdpa"
    revision: str | None = None
    local_files_only: bool = False
    use_embedded_vision_weights: bool = True
    warmup: bool = True

    def __post_init__(self) -> None:
        root = Path(self.streamvln_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"StreamVLN root does not exist: {root}")
        object.__setattr__(self, "streamvln_root", root)
        object.__setattr__(self, "model_path", _checkpoint_name(self.model_path))
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        object.__setattr__(self, "device", self.device.strip())
        object.__setattr__(
            self,
            "cuda_memory_fraction",
            _cuda_fraction(self.cuda_memory_fraction),
        )
        for name in (
            "num_future_steps",
            "num_frames",
            "num_history",
            "model_max_length",
            "max_new_tokens",
        ):
            _positive_int(getattr(self, name), name)
        if self.num_future_steps > MAX_FUTURE_ACTIONS:
            raise ValueError(f"num_future_steps must not exceed {MAX_FUTURE_ACTIONS}")
        if self.dtype != "bfloat16":
            raise ValueError("StreamVLN currently supports only dtype='bfloat16'")
        if self.attn_implementation is not None and (
            not isinstance(self.attn_implementation, str) or not self.attn_implementation.strip()
        ):
            raise ValueError("attn_implementation must be None or a non-empty string")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ValueError("revision must be None or a non-empty string")
        if not isinstance(self.local_files_only, bool):
            raise ValueError("local_files_only must be a boolean")  # noqa: TRY004
        if not isinstance(self.use_embedded_vision_weights, bool):
            raise ValueError(  # noqa: TRY004
                "use_embedded_vision_weights must be a boolean"
            )
        if not isinstance(self.warmup, bool):
            raise ValueError("warmup must be a boolean")  # noqa: TRY004


__all__ = ["DEFAULT_MODEL", "DEFAULT_SENSOR_CONFIG", "StreamVLNConfig"]
