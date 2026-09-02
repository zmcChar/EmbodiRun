"""Service planning and supervision errors."""


class ServiceError(ValueError):
    """A service cannot be constructed or supervised safely."""


__all__ = ["ServiceError"]
