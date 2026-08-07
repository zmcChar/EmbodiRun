"""Lazy LeRobot pi0.5 dependency loading."""

from __future__ import annotations

from ...errors import ModelPackageError


def require_lerobot():
    try:
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
    except ImportError as error:
        raise ModelPackageError(
            "pi0.5 requires lerobot==0.5.1 plus its transformers dependency; "
            "install the project's 'pi05' extra"
        ) from error
    return (
        PI05Config,
        PI05Policy,
        PolicyFeature,
        FeatureType,
        NormalizationMode,
        RTCConfig,
    )
