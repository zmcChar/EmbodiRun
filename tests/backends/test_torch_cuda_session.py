"""PyTorch session helper tests that use only fake tensor values."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from embodied_runtime.backends.torch_cuda.compiler import TorchArtifactPayload
from embodied_runtime.backends.torch_cuda.session_stage import (
    actual_execution_mode,
    invoke_entrypoint,
)
from embodied_runtime.backends.torch_cuda.session_transfer import (
    add_scaled_tree,
    move_tensor_tree,
)


def test_artifact_session_copy_and_deferred_fallback_are_private() -> None:
    eager = lambda value: value + 1

    def compiled(value):
        raise RuntimeError("deferred failure")

    artifact_payload = TorchArtifactPayload(
        package_id="package",
        entrypoints={"forward": compiled},
        eager_entrypoints={"forward": eager},
        device_id="cpu",
        dtype=None,
        requested_mode="compile",
        actual_mode="compile",
        fallback_to_eager=True,
    )
    session_payload = artifact_payload.for_session()

    assert invoke_entrypoint(session_payload, "forward", 2) == 3
    assert actual_execution_mode(session_payload) == "eager"
    assert "deferred RuntimeError" in session_payload.compile_failures["forward"]
    assert artifact_payload.entrypoints["forward"] is compiled
    assert artifact_payload.compile_failures == {}


@dataclass(frozen=True)
class _Pair:
    left: object
    right: object


class _FakeTensor:
    def __init__(self, value: float, *, floating: bool = True) -> None:
        self.value = value
        self.floating = floating
        self.moves: list[dict[str, object]] = []

    def is_floating_point(self) -> bool:
        return self.floating

    def to(self, **kwargs):
        self.moves.append(kwargs)
        return self


class _TreeTorch:
    Tensor = _FakeTensor


def test_session_transfer_and_state_math_preserve_tree_semantics() -> None:
    floating = _FakeTensor(1.0)
    integer = _FakeTensor(2.0, floating=False)
    tree = _Pair(left={"tensor": floating}, right=[integer])

    moved = move_tensor_tree(
        _TreeTorch(),
        tree,
        device="cuda:0",
        dtype="float16",
        non_blocking=True,
    )

    assert moved is not tree
    assert floating.moves == [{"device": "cuda:0", "dtype": "float16", "non_blocking": True}]
    assert integer.moves == [{"device": "cuda:0", "dtype": None, "non_blocking": True}]
    assert add_scaled_tree(_TreeTorch(), {"x": [1.0]}, {"x": [2.0]}, 0.25) == {"x": [1.5]}
    with pytest.raises(TypeError, match="identical keys"):
        add_scaled_tree(_TreeTorch(), {"x": 1.0}, {"y": 1.0}, 1.0)
