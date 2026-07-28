"""RLinf-compatible OpenVLA-OFT VLA adapter."""

from .adapter import OpenVLAOFTAdapter
from .head import CategoricalActionHead
from .modeling_openvla_oft import (
    OpenVLAOFTReferenceModule,
    load_openvla_oft,
    resolve_openvla_oft_checkpoint,
)
from .processing_openvla_oft import (
    OpenVLAOFTProcessor,
    PROMPT_TEMPLATE,
    SPACE_TOKEN,
)

__all__ = [
    "CategoricalActionHead",
    "OpenVLAOFTAdapter",
    "OpenVLAOFTProcessor",
    "OpenVLAOFTReferenceModule",
    "PROMPT_TEMPLATE",
    "SPACE_TOKEN",
    "load_openvla_oft",
    "resolve_openvla_oft_checkpoint",
]
