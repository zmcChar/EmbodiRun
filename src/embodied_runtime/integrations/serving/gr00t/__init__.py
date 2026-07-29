"""GR00T N1.7 provider endpoints, imported without cross-provider coupling."""

from __future__ import annotations

from typing import Any

__all__ = ["HfLocalGr00tProvider", "VllmOmniGr00tProvider"]


def __getattr__(name: str) -> Any:
    if name == "HfLocalGr00tProvider":
        from .hf_local import HfLocalGr00tProvider

        return HfLocalGr00tProvider
    if name == "VllmOmniGr00tProvider":
        from .vllm_omni import VllmOmniGr00tProvider

        return VllmOmniGr00tProvider
    raise AttributeError(name)
