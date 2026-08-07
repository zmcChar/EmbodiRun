"""Extraction and validation of structured planner request data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.tasks.planning import PlanRequest

from .errors import PlannerInputError
from .values import TexasHoldemCard

METADATA_NAMESPACE = "texas_holdem"


def cards_from_request(request: PlanRequest) -> tuple[TexasHoldemCard, ...]:
    sources: list[tuple[str, Any]] = []
    goal_section = request.goal.metadata.get(METADATA_NAMESPACE)
    if goal_section is not None:
        sources.append(("goal.metadata", goal_section))
    observation = request.observation
    if isinstance(observation, Mapping):
        observation_metadata = observation.get("metadata")
        if isinstance(observation_metadata, Mapping):
            observation_section = observation_metadata.get(METADATA_NAMESPACE)
            if observation_section is not None:
                sources.append(("observation metadata", observation_section))
    if not sources:
        raise PlannerInputError(
            "missing texas_holdem metadata in goal.metadata or observation metadata"
        )
    parsed = tuple((name, cards_from_metadata(value, source=name)) for name, value in sources)
    reference = parsed[0][1]
    for source_name, cards in parsed[1:]:
        if cards != reference:
            raise PlannerInputError(f"{source_name} cards conflict with {parsed[0][0]} cards")
    return reference


def cards_from_metadata(value: Any, *, source: str) -> tuple[TexasHoldemCard, ...]:
    if not isinstance(value, Mapping):
        raise PlannerInputError(f"{source}.{METADATA_NAMESPACE} must be a mapping")
    if set(value) != {"cards"}:
        raise PlannerInputError(
            f"{source}.{METADATA_NAMESPACE} must contain exactly the 'cards' key"
        )
    raw_cards = value["cards"]
    if isinstance(raw_cards, (str, bytes)) or not isinstance(raw_cards, Sequence):
        raise PlannerInputError(f"{source} cards must be a sequence")
    cards: list[TexasHoldemCard] = []
    for index, raw_card in enumerate(raw_cards):
        if not isinstance(raw_card, Mapping):
            raise PlannerInputError(f"{source} card {index} must be a mapping")
        if set(raw_card) != {"name", "value", "suit"}:
            raise PlannerInputError(
                f"{source} card {index} must contain exactly name, value, and suit"
            )
        cards.append(
            TexasHoldemCard(
                name=raw_card["name"],
                value=raw_card["value"],
                suit=raw_card["suit"],
            )
        )
    return validate_cards(cards)


def validate_cards(cards: Sequence[TexasHoldemCard]) -> tuple[TexasHoldemCard, ...]:
    if isinstance(cards, (str, bytes)) or not isinstance(cards, Sequence):
        raise PlannerInputError("cards must be a sequence of TexasHoldemCard values")
    normalized = tuple(cards)
    if len(normalized) < 5:
        raise PlannerInputError("Texas Hold'em planning requires at least five cards")
    if any(not isinstance(card, TexasHoldemCard) for card in normalized):
        raise PlannerInputError("cards must contain only TexasHoldemCard values")
    names = tuple(card.name for card in normalized)
    if len(set(names)) != len(names):
        raise PlannerInputError("input card names must be unique")
    physical_cards = tuple((card.value, card.suit) for card in normalized)
    if len(set(physical_cards)) != len(physical_cards):
        raise PlannerInputError("input value/suit card pairs must be unique")
    return normalized
