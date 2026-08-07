"""Strict parsing and grounding of Texas Hold'em model output."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from .errors import PlannerOutputError
from .request_data import validate_cards
from .values import (
    SUPPORTED_HAND_TYPES,
    TARGET_COUNT_BY_HAND_TYPE,
    TexasHoldemCard,
    TexasHoldemSelection,
)

_THINK_THEN_JSON = re.compile(r"\A<think>.*?</think>\s*(\{.*\})\Z", re.DOTALL)
_SINGLE_JSON_FENCE = re.compile(r"\A```json\s*(\{.*\})\s*```\Z", re.DOTALL)


def parse_texas_holdem_selection(
    response: str,
    *,
    cards: Sequence[TexasHoldemCard],
    allow_single_json_fence: bool = False,
) -> TexasHoldemSelection:
    """Parse strict JSON and enforce grounding to the input card set."""

    if not isinstance(response, str) or not response.strip():
        raise PlannerOutputError("planner response must be a non-empty string")
    if not isinstance(allow_single_json_fence, bool):
        raise TypeError("allow_single_json_fence must be a bool")
    normalized_cards = validate_cards(cards)
    payload_text = response.strip()
    think_match = _THINK_THEN_JSON.fullmatch(payload_text)
    if think_match is not None:
        payload_text = think_match.group(1)
    elif allow_single_json_fence:
        fence_match = _SINGLE_JSON_FENCE.fullmatch(payload_text)
        if fence_match is not None:
            payload_text = fence_match.group(1)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise PlannerOutputError("planner response is not strict JSON") from error
    if not isinstance(payload, Mapping):
        raise PlannerOutputError("planner response must be a JSON object")
    expected_keys = {"hand_type", "target_card_names"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys.difference(payload))
        extra = sorted(set(payload).difference(expected_keys))
        raise PlannerOutputError(
            f"planner response schema mismatch; missing={missing}, extra={extra}"
        )
    hand_type = payload["hand_type"]
    if not isinstance(hand_type, str) or hand_type not in SUPPORTED_HAND_TYPES:
        raise PlannerOutputError(f"unsupported hand_type: {hand_type!r}")
    raw_names = payload["target_card_names"]
    if isinstance(raw_names, (str, bytes)) or not isinstance(raw_names, Sequence):
        raise PlannerOutputError("target_card_names must be a JSON array")
    names = tuple(raw_names)
    if not names:
        raise PlannerOutputError("target_card_names must not be empty")
    if len(names) > 5:
        raise PlannerOutputError("target_card_names must contain at most five cards")
    if any(not isinstance(name, str) or not name for name in names):
        raise PlannerOutputError("every target_card_names item must be a non-empty string")
    if len(set(names)) != len(names):
        raise PlannerOutputError("target_card_names must not contain duplicates")
    expected_count = TARGET_COUNT_BY_HAND_TYPE[hand_type]
    if len(names) != expected_count:
        raise PlannerOutputError(
            f"hand_type {hand_type!r} requires exactly {expected_count} target cards"
        )
    allowed_names = {card.name for card in normalized_cards}
    unknown = sorted(set(names).difference(allowed_names))
    if unknown:
        raise PlannerOutputError(f"planner selected cards absent from the input: {unknown}")
    return TexasHoldemSelection(hand_type=hand_type, target_card_names=names)


__all__ = ["parse_texas_holdem_selection"]
