"""Robot-scoped identities for multi-tenant cloud inference."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _validate_identifier(name: str, value: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            f"{name} must be 1-128 characters using letters, digits, '.', '_', ':', or '-'"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class RobotSessionIdentity:
    """Identity of one robot's model session.

    ``robot_id`` survives process and episode restarts. ``edge_node_id`` names
    one logical edge runtime. ``session_id`` isolates one episode or stateful
    policy session and should be rotated when that session is reset.
    """

    robot_id: str
    edge_node_id: str
    session_id: str
    embodiment: str
    action_space_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "robot_id",
            _validate_identifier("robot_id", self.robot_id),
        )
        object.__setattr__(
            self,
            "edge_node_id",
            _validate_identifier("edge_node_id", self.edge_node_id),
        )
        object.__setattr__(
            self,
            "session_id",
            _validate_identifier("session_id", self.session_id),
        )
        object.__setattr__(
            self,
            "embodiment",
            _validate_identifier("embodiment", self.embodiment),
        )
        object.__setattr__(
            self,
            "action_space_id",
            _validate_identifier("action_space_id", self.action_space_id),
        )

    def to_wire(self) -> dict[str, str]:
        return {
            "robot_id": self.robot_id,
            "edge_node_id": self.edge_node_id,
            "session_id": self.session_id,
            "embodiment": self.embodiment,
            "action_space_id": self.action_space_id,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> RobotSessionIdentity:
        names = (
            "robot_id",
            "edge_node_id",
            "session_id",
            "embodiment",
            "action_space_id",
        )
        try:
            values = {name: payload[name] for name in names}
        except KeyError as error:
            raise ValueError(f"session identity is missing {error.args[0]!r}") from error
        for name, value in values.items():
            if not isinstance(value, str):
                raise TypeError(f"session identity {name} must be a string")
        return cls(**values)

    def namespace_request_id(self, request_id: str) -> str:
        """Create an engine-global ID without conflating robot sessions."""

        external = _validate_identifier("request_id", request_id)
        return f"{len(self.session_id)}:{self.session_id}:{external}"


__all__ = ["RobotSessionIdentity"]
