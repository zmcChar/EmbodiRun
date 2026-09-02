"""Environment preparation errors."""


class EnvironmentError(ValueError):
    """The requested environment layout is ambiguous or unsupported."""


__all__ = ["EnvironmentError"]
