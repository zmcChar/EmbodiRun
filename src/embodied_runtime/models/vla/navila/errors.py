"""Stable model-runtime errors for NaVILA."""


class NaVILAError(RuntimeError):
    """Base class for NaVILA runtime failures."""


class NaVILALoadError(NaVILAError):
    """The official source or checkpoint could not be loaded."""


class NaVILAInferenceError(NaVILAError):
    """A loaded model failed to produce a usable prediction."""


class NaVILANativeOutputError(ValueError):
    """Decoded model text is not one valid NaVILA navigation action."""


__all__ = [
    "NaVILAError",
    "NaVILAInferenceError",
    "NaVILALoadError",
    "NaVILANativeOutputError",
]
