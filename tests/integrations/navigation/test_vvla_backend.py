from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodied_runtime.integrations.navigation.vvla.backend import ValidatedActiveVLNBackend
from embodied_runtime.policies.navigation.errors import NavigationPolicyError


def _action(name: str, value: int | None) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


def _chunk(text: str, actions: tuple[SimpleNamespace, ...], *, valid: bool = True):
    parsed = SimpleNamespace(valid=valid, raw_text=text, actions=actions)
    trace = SimpleNamespace(text=text, parsed_actions=parsed)
    return SimpleNamespace(trace=trace)


class _Backend:
    def __init__(self, chunk) -> None:
        self.chunk = chunk
        self.resets = []
        self.cancels = []

    def generate(self, observations, *, session_ids):
        return [self.chunk]

    def reset_sessions(self, session_ids) -> None:
        self.resets.append(session_ids)

    def cancel_sessions(self, session_ids) -> None:
        self.cancels.append(session_ids)


def test_validated_backend_accepts_only_canonical_action_text() -> None:
    chunk = _chunk("move forward 25cm", (_action("move forward", 25),))
    backend = ValidatedActiveVLNBackend(_Backend(chunk))

    assert backend.generate([object()], session_ids=["episode"]) == [chunk]


@pytest.mark.parametrize(
    ("text", "actions", "message"),
    [
        ("please do not move forward 25cm", (_action("move forward", 25),), "non-canonical"),
        ("move forward 25cm", (_action("turn left", 15),), "do not match"),
        (
            "stop, move forward 25cm",
            (_action("stop", None), _action("move forward", 25)),
            "final action",
        ),
    ],
)
def test_validated_backend_rejects_outputs_the_upstream_parser_accepts(
    text: str,
    actions: tuple[SimpleNamespace, ...],
    message: str,
) -> None:
    backend = ValidatedActiveVLNBackend(_Backend(_chunk(text, actions)))

    with pytest.raises(NavigationPolicyError, match=message):
        backend.generate([object()], session_ids=["episode"])


def test_validated_backend_checks_decoder_text_and_forwards_session_controls() -> None:
    chunk = _chunk("stop", (_action("stop", None),))
    chunk.trace.text = "move forward 25cm"
    raw = _Backend(chunk)
    backend = ValidatedActiveVLNBackend(raw)

    with pytest.raises(NavigationPolicyError, match="decoder text"):
        backend.generate([object()], session_ids=["episode"])
    backend.reset_sessions(["episode"])
    backend.cancel_sessions(["episode"])

    assert raw.resets == [["episode"]]
    assert raw.cancels == [["episode"]]
