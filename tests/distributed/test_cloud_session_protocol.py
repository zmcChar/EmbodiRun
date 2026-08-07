from __future__ import annotations

import math
from enum import Enum

import pytest

from embodied_runtime.distributed.communication.cloud_session import (
    CloudSessionRemoteError,
    json_compatible,
)


class _Mode(Enum):
    READY = "ready"


class _TensorLike:
    def detach(self) -> _TensorLike:
        return self

    def cpu(self) -> _TensorLike:
        return self

    def tolist(self) -> list[float]:
        return [1.0, 2.0]


def test_json_compatible_normalizes_nested_enum_and_tensor_values() -> None:
    assert json_compatible(
        {
            "mode": _Mode.READY,
            "tensor": _TensorLike(),
            "nested": (True, 3),
        }
    ) == {
        "mode": "ready",
        "tensor": [1.0, 2.0],
        "nested": [True, 3],
    }


def test_json_compatible_preserves_precise_error_paths() -> None:
    with pytest.raises(ValueError, match=r"value\.samples\[1\].*non-finite"):
        json_compatible({"samples": [1.0, math.inf]})
    with pytest.raises(TypeError, match="non-string mapping key"):
        json_compatible({1: "invalid"})


def test_cloud_session_remote_error_retains_remote_details() -> None:
    error = CloudSessionRemoteError("SessionRegistrationError", "unknown session")

    assert error.error_type == "SessionRegistrationError"
    assert error.remote_message == "unknown session"
    assert str(error) == "SessionRegistrationError: unknown session"
