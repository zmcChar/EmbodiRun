"""Compiled model artifact contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import ModelPackage
from .resource import DeviceInfo
from .types import Metadata


@dataclass(frozen=True, slots=True)
class CompileOptions:
    mode: str = "eager"
    dtype: str | None = None
    dynamic_shapes: bool = False
    options: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactVariant:
    backend: str
    device: DeviceInfo
    package: ModelPackage
    options: CompileOptions
    payload: Any
    metadata: Metadata = field(default_factory=dict)
