"""Errors raised at the Qwen navigation boundary."""


class QwenNavigationError(RuntimeError):
    """A Qwen request or response cannot satisfy the navigation contract."""


class QwenNavigationValidationError(QwenNavigationError):
    """A model response is not a valid shared waypoint plan."""


__all__ = ["QwenNavigationError", "QwenNavigationValidationError"]
