"""Fixed-size UDP framing for wired SO-101 teleoperation.

The packet is deliberately minimal: a magic word, a sequence number and the six
joint targets. It is sent at the control rate, so anything larger would cost
bandwidth for no benefit. The sequence number lets a follower drop duplicates
and detect a stalled stream.
"""

from __future__ import annotations

import struct

MAGIC = b"S101"
JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
PACKET = struct.Struct("!4sI6f")


class ProtocolError(ValueError):
    """A datagram is not a well-formed teleoperation packet."""


def encode(sequence: int, action: dict[str, float]) -> bytes:
    """Serialise ``action`` for the given sequence number."""

    try:
        values = [float(action[name]) for name in JOINTS]
    except KeyError as error:
        raise ProtocolError(f"action is missing joint {error}") from error
    return PACKET.pack(MAGIC, sequence & 0xFFFFFFFF, *values)


def decode(data: bytes) -> tuple[int, dict[str, float]]:
    """Parse a datagram into ``(sequence, action)``."""

    if len(data) != PACKET.size:
        raise ProtocolError(f"wrong packet size: {len(data)} != {PACKET.size}")
    magic, sequence, *values = PACKET.unpack(data)
    if magic != MAGIC:
        raise ProtocolError(f"wrong packet magic: {magic!r}")
    return sequence, dict(zip(JOINTS, values, strict=True))
