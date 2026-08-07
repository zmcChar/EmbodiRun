"""Lazy dependency checks for OpenVLA-OFT."""

from __future__ import annotations

from ...errors import ModelPackageError


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise ModelPackageError(
            "OpenVLA-OFT requires PyTorch; install the 'torch' and 'openvla_oft' extras"
        ) from error
    return torch
