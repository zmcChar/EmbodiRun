"""Lazy, version-pinned SmolVLA dependency loading."""

from __future__ import annotations

import importlib.metadata

from ...errors import ModelPackageError
from .constants import EXPECTED_LEROBOT_VERSION


def require_lerobot_033():
    try:
        installed = importlib.metadata.version("lerobot")
    except importlib.metadata.PackageNotFoundError as error:
        raise ModelPackageError(
            "SmolVLA requires the isolated 'smolvla' extra with lerobot==0.3.3"
        ) from error
    if installed != EXPECTED_LEROBOT_VERSION:
        raise ModelPackageError(
            "the legacy SmolVLA adapter requires lerobot=="
            f"{EXPECTED_LEROBOT_VERSION}, found {installed}; keep it in an "
            "endpoint environment separate from the pi0.5 runtime"
        )
    try:
        # Importing the policy registers its config subclass before generic
        # PreTrainedConfig resolution.
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError as error:
        raise ModelPackageError(
            "SmolVLA requires lerobot[smolvla]==0.3.3 and transformers"
        ) from error
    return PreTrainedConfig, SmolVLAPolicy
