"""Compatibility exports for the split GR00T provider modules.

New code may import ``hf_local`` or ``vllm_omni`` directly.  Lazy attribute
resolution keeps the remote vLLM-Omni path independent of local Backend code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .hf_local import HfLocalGr00tProvider
    from .vllm_omni import VllmOmniGr00tProvider

__all__ = ["HfLocalGr00tProvider", "VllmOmniGr00tProvider"]


def __getattr__(name: str) -> Any:
    if name == "HfLocalGr00tProvider":
        from .hf_local import HfLocalGr00tProvider

        return HfLocalGr00tProvider
    if name == "VllmOmniGr00tProvider":
        from .vllm_omni import VllmOmniGr00tProvider

        return VllmOmniGr00tProvider
    raise AttributeError(name)
