"""Experimental vLLM-Omni/OpenPI navigation clients and wire contract."""

from __future__ import annotations

from typing import Any

__all__ = [
    "NAVIGATION_PROTOCOL_NAME",
    "NAVIGATION_PROTOCOL_VERSION",
    "VllmOmniNaVILANavigationPolicy",
    "VllmOmniStreamVLNNavigationPolicy",
    "decode_navila_action",
    "decode_streamvln_actions",
    "encode_navila_action",
    "encode_streamvln_actions",
    "navigation_handshake_metadata",
]


def __getattr__(name: str) -> Any:
    if name == "VllmOmniStreamVLNNavigationPolicy":
        from .streamvln import VllmOmniStreamVLNNavigationPolicy

        return VllmOmniStreamVLNNavigationPolicy
    if name == "VllmOmniNaVILANavigationPolicy":
        from .navila import VllmOmniNaVILANavigationPolicy

        return VllmOmniNaVILANavigationPolicy
    if name in {
        "decode_navila_action",
        "decode_streamvln_actions",
        "encode_navila_action",
        "encode_streamvln_actions",
    }:
        from . import codec

        return getattr(codec, name)
    if name in {
        "NAVIGATION_PROTOCOL_NAME",
        "NAVIGATION_PROTOCOL_VERSION",
        "navigation_handshake_metadata",
    }:
        from . import common

        return getattr(common, name)
    raise AttributeError(name)
