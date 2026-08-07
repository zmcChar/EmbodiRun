"""RLinf-compatible OpenVLA-OFT VLA adapter."""

from typing import Any

from .adapter import OpenVLAOFTAdapter
from .checkpoint import resolve_openvla_oft_checkpoint
from .constants import PROMPT_TEMPLATE, SPACE_TOKEN
from .loading import load_openvla_oft


def __getattr__(name: str) -> Any:
    """Load Torch-dependent implementation classes only when requested."""

    if name == "CategoricalActionHead":
        from .head import CategoricalActionHead

        return CategoricalActionHead
    if name == "OpenVLAOFTReferenceModule":
        from .modeling_openvla_oft import OpenVLAOFTReferenceModule

        return OpenVLAOFTReferenceModule
    if name == "OpenVLAOFTProcessor":
        from .processing_openvla_oft import OpenVLAOFTProcessor

        return OpenVLAOFTProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PROMPT_TEMPLATE",
    "SPACE_TOKEN",
    "CategoricalActionHead",
    "OpenVLAOFTAdapter",
    "OpenVLAOFTProcessor",
    "OpenVLAOFTReferenceModule",
    "load_openvla_oft",
    "resolve_openvla_oft_checkpoint",
]
