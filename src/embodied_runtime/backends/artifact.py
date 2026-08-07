"""Compiled model artifact owned by a hardware backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from embodied_runtime.models.package import ModelPackage
from embodied_runtime.types import Metadata

from .compile import CompileOptions
from .device import DeviceInfo


@dataclass(slots=True)
class ArtifactVariant:
    backend: str
    device: DeviceInfo
    package: ModelPackage
    options: CompileOptions
    payload: Any
    metadata: Metadata = field(default_factory=dict)


__all__ = ["ArtifactVariant"]
