"""Root exception shared by independently owned runtime domains."""


class EmbodiedRuntimeError(Exception):
    """Base class for errors intended to cross a domain boundary."""


__all__ = ["EmbodiedRuntimeError"]
