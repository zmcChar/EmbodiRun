"""Privileged state inspection used only to validate paired VLABench deals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .vlabench_privileged_values import TexasHoldemDealIdentity


def inspect_texas_holdem_deal(endpoint: Any) -> TexasHoldemDealIdentity:
    """Inspect privileged Texas Hold'em identity for pairing validation only."""

    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    if task is None:
        raise RuntimeError("VLABench endpoint did not expose a live Texas Hold'em task")
    pokers = tuple(getattr(task, "pokers", ()))
    if not pokers:
        raise RuntimeError("Texas Hold'em task did not expose task.pokers")
    card_names = tuple(str(getattr(poker, "name", "")) for poker in pokers)
    raw_targets = getattr(task, "target_entities", None)
    if isinstance(raw_targets, Mapping) or (
        isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes))
    ):
        target_names = tuple(str(name) for name in raw_targets)
    else:
        raise TypeError("Texas Hold'em task did not expose target_entities as names")
    return TexasHoldemDealIdentity(
        card_names=card_names,
        target_card_names=target_names,
        hand_type=str(getattr(task, "max_cardtype", "")),
    )


def endpoint_fingerprint(endpoint: Any) -> str:
    fingerprint = getattr(endpoint, "initial_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise RuntimeError("VLABench endpoint must be reset before oracle pairing")
    return fingerprint.strip()


def validate_endpoint_seed_if_available(endpoint: Any, expected_seed: int) -> None:
    observe = getattr(endpoint, "observe", None)
    if not callable(observe):
        return
    observation = observe()
    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return
    actual_seed = metadata.get("seed")
    if actual_seed is not None and actual_seed != expected_seed:
        raise RuntimeError(
            f"VLABench endpoint was reset with seed {actual_seed}, expected {expected_seed}"
        )


def validate_selected_cards(
    selected_cards: tuple[str, ...],
    deal: TexasHoldemDealIdentity,
) -> None:
    missing = set(selected_cards).difference(deal.card_names)
    if missing:
        raise RuntimeError(
            f"selected cards are missing from the paired poker deal: {sorted(missing)}"
        )


__all__ = ["inspect_texas_holdem_deal"]
