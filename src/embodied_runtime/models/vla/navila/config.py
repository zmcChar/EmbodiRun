"""Validated configuration for the official NaVILA navigation checkpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "a8cheng/navila-llama3-8b-8f"
REFERENCE_SOURCE_COMMIT = "76b98f233dd0fff05dfcd69435eec6740febff9d"


def _checkpoint(value: str | Path) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("model_path must be a non-empty path or model id")
    candidate = Path(value).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("a local NaVILA model_path must be a checkpoint directory")
        return str(candidate.resolve())
    return str(value).strip()


def _cuda_fraction(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cuda_memory_fraction must be a number or None")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError("cuda_memory_fraction must be in (0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class NaVILAConfig:
    """Settings for one local, lazy NaVILA model runtime."""

    navila_root: str | Path
    model_path: str | Path = DEFAULT_MODEL
    device: str = "cuda:0"
    cuda_memory_fraction: float | None = None
    max_new_tokens: int = 32
    dtype: str = "float16"
    attention_backend: str = "eager"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        root = Path(self.navila_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"NaVILA root does not exist: {root}")
        for relative in ("llava/constants.py", "llava/mm_utils.py", "llava/model/builder.py"):
            if not (root / relative).is_file():
                raise ValueError(f"NaVILA root is missing {relative}")
        object.__setattr__(self, "navila_root", root)
        object.__setattr__(self, "model_path", _checkpoint(self.model_path))
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty torch device string")
        device = self.device.strip()
        if device != "cuda" and not (
            device.startswith("cuda:") and device.removeprefix("cuda:").isdigit()
        ):
            raise ValueError("NaVILA requires device='cuda' or 'cuda:<index>'")
        object.__setattr__(self, "device", device)
        object.__setattr__(
            self,
            "cuda_memory_fraction",
            _cuda_fraction(self.cuda_memory_fraction),
        )
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be a positive integer")
        if not 1 <= self.max_new_tokens <= 128:
            raise ValueError("max_new_tokens must be between 1 and 128")
        if self.dtype != "float16":
            raise ValueError("NaVILA currently supports only dtype='float16'")
        if self.attention_backend != "eager":
            raise ValueError("NaVILA currently supports only attention_backend='eager'")
        if not isinstance(self.local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")


__all__ = ["DEFAULT_MODEL", "REFERENCE_SOURCE_COMMIT", "NaVILAConfig"]
