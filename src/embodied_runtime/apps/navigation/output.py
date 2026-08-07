"""JSON-lines output for navigation CLI events and summaries."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path


def json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def emit(output: Callable[[str], None], payload: Mapping[str, object]) -> None:
    output(json.dumps(json_value(payload), ensure_ascii=False, sort_keys=True))


__all__ = ["emit", "json_value"]
