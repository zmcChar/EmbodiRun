"""Errors raised by multi-tenant inference session management."""


class SessionRegistrationError(RuntimeError):
    """A robot session is missing or conflicts with an existing registration."""


class SessionSequenceError(RuntimeError):
    """A request is duplicated or arrives out of order within one session."""


class SessionOverloadedError(RuntimeError):
    """A session already has the maximum allowed pending requests."""


__all__ = ["SessionOverloadedError", "SessionRegistrationError", "SessionSequenceError"]
