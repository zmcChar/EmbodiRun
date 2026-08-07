"""Task values shared by prompt construction, parsing, and planning."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import PlannerInputError

SUPPORTED_HAND_TYPES = frozenset(
    {
        "royal_flush",
        "straight_flush",
        "four_of_a_kind",
        "full_house",
        "flush",
        "straight",
        "three_of_a_kind",
        "two_pair",
        "one_pair",
        "high_card",
    }
)
TARGET_COUNT_BY_HAND_TYPE = {
    "royal_flush": 5,
    "straight_flush": 5,
    "four_of_a_kind": 4,
    "full_house": 5,
    "flush": 5,
    "straight": 5,
    "three_of_a_kind": 3,
    "two_pair": 4,
    "one_pair": 2,
    "high_card": 1,
}


@dataclass(frozen=True, slots=True)
class TexasHoldemCard:
    """One visible card supplied to the planner without simulator references."""

    name: str
    value: str
    suit: str

    def __post_init__(self) -> None:
        for field_name in ("name", "value", "suit"):
            raw = getattr(self, field_name)
            if not isinstance(raw, str):
                raise PlannerInputError(f"card {field_name} must be a string")
            normalized = raw.strip()
            if not normalized:
                raise PlannerInputError(f"card {field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

    @property
    def primitive_instruction(self) -> str:
        return f"primitive: Please pick the poker {self.value} of {self.suit}"


@dataclass(frozen=True, slots=True)
class TexasHoldemSelection:
    """Validated model selection grounded to the supplied card names."""

    hand_type: str
    target_card_names: tuple[str, ...]


__all__ = ["TexasHoldemCard"]
