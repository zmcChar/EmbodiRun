"""Safety boundary for ActiveVLN generation backends."""

from __future__ import annotations

from typing import Any

from embodied_runtime.policies.navigation.errors import NavigationPolicyError

from .mapping import validate_activevln_action_text


def validate_activevln_chunk(chunk: Any) -> Any:
    """Validate one decoder chunk and return its typed navigation parse."""

    trace = getattr(chunk, "trace", None)
    parsed = None if trace is None else getattr(trace, "parsed_actions", None)
    raw_text = "" if trace is None or trace.text is None else trace.text
    if parsed is None or not parsed.valid:
        raise NavigationPolicyError(
            f"VVLA ActiveVLN returned an invalid action response: {raw_text!r}"
        )
    if parsed.raw_text != raw_text:
        raise NavigationPolicyError(
            "VVLA ActiveVLN decoder text does not match its parsed action text"
        )
    try:
        validate_activevln_action_text(raw_text, parsed.actions)
    except ValueError as error:
        raise NavigationPolicyError(str(error)) from error
    return parsed


class ValidatedActiveVLNBackend:
    """Transparent backend proxy that rejects unsafe text before consumers act."""

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        chunks = self._backend.generate(*args, **kwargs)
        for chunk in chunks:
            validate_activevln_chunk(chunk)
        return chunks

    def reset_sessions(self, session_ids: Any) -> None:
        self._backend.reset_sessions(session_ids)

    def cancel_sessions(self, session_ids: Any) -> None:
        self._backend.cancel_sessions(session_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


__all__ = ["ValidatedActiveVLNBackend", "validate_activevln_chunk"]
