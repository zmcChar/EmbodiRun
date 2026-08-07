"""Errors raised by navigation-policy adapters."""


class NavigationPolicyError(RuntimeError):
    """A navigation policy cannot safely advance its model state."""


__all__ = ["NavigationPolicyError"]
