"""Prompt construction for structured Texas Hold'em planning."""

from __future__ import annotations

import json
from collections.abc import Sequence

from .errors import PlannerInputError
from .request_data import validate_cards
from .values import SUPPORTED_HAND_TYPES, TexasHoldemCard


def build_texas_holdem_prompt(
    *,
    instruction: str,
    cards: Sequence[TexasHoldemCard],
) -> str:
    """Build an oracle-comparable prompt using VLABench's target-card rules."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise PlannerInputError("instruction must be a non-empty string")
    normalized_cards = validate_cards(cards)
    card_payload = [
        {"name": card.name, "value": card.value, "suit": card.suit} for card in normalized_cards
    ]
    hand_types = ", ".join(sorted(SUPPORTED_HAND_TYPES))
    return (
        "Solve this Texas Hold'em card-selection task from the structured card list.\n"
        f"Task instruction: {instruction.strip()}\n"
        f"Cards in input order: {json.dumps(card_payload, ensure_ascii=True)}\n\n"
        "Use standard five-card poker category strength. Evaluate every five-card "
        "combination in input order and use the first combination that reaches the "
        "strongest category. Return only the category-defining cards: one card for "
        "high_card, two for one_pair, four for two_pair, three for three_of_a_kind, "
        "four for four_of_a_kind, and all five for straight, flush, full_house, "
        "straight_flush, or royal_flush. Every target must copy a card name exactly "
        "from the input.\n\n"
        "Return exactly one JSON object and no Markdown or explanation. The object "
        "must contain exactly these keys:\n"
        '{"hand_type":"<category>","target_card_names":["<input card name>",...]}\n'
        f"hand_type must be one of: {hand_types}."
    )


__all__ = ["build_texas_holdem_prompt"]
