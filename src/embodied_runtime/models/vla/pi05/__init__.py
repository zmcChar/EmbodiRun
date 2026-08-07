"""Canonical pi0.5 VLA adapter package."""

from typing import Any

from .adapter import Pi05Adapter, load_lerobot_pi05


def __getattr__(name: str) -> Any:
    """Load Torch-dependent implementation details only when requested."""

    if name == "Pi05ReferenceModule":
        from .modeling_pi05 import Pi05ReferenceModule

        return Pi05ReferenceModule
    if name in {"LANGUAGE_ATTENTION_MASK", "LANGUAGE_TOKENS", "Pi05Processor"}:
        from . import processing_pi05

        return getattr(processing_pi05, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LANGUAGE_ATTENTION_MASK",
    "LANGUAGE_TOKENS",
    "Pi05Adapter",
    "Pi05Processor",
    "Pi05ReferenceModule",
    "load_lerobot_pi05",
]
