"""Tests for supported-Python compatibility helpers."""

from __future__ import annotations

from enum import auto

from embodied_runtime._compat import StrEnum


class _ExplicitValue(StrEnum):
    READY = "ready"


class _GeneratedValue(StrEnum):
    CLOUD_NODE = auto()


def test_strenum_matches_required_stdlib_semantics() -> None:
    assert _ExplicitValue.READY == "ready"
    assert str(_ExplicitValue.READY) == "ready"
    assert _GeneratedValue.CLOUD_NODE.value == "cloud_node"
