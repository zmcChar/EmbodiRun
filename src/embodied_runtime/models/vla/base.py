"""Shared VLA result semantics layered on the generic model adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_runtime.types import TensorTree

from ..action import ActionChunk
from ..base import BaseModelAdapter
from ..errors import ModelPackageError
from ..request import RawRequest


def _actions_from(output: TensorTree) -> Any:
    if isinstance(output, Mapping):
        try:
            return output["actions"]
        except KeyError as exc:
            raise ModelPackageError("VLA output mapping must contain 'actions'") from exc
    return output


class VLAAdapterBase(BaseModelAdapter[RawRequest, ActionChunk]):
    """Base for models whose semantic result is an action chunk.

    Device placement, scheduling, compilation, and robot SDK conversion remain
    outside this class.
    """

    def postprocess_one(self, output: TensorTree) -> ActionChunk:
        actions = _actions_from(output)
        ndim = getattr(actions, "ndim", None)
        if ndim is None:
            raise ModelPackageError("VLA actions must be tensor-like")
        if ndim >= 3:
            raise ModelPackageError(
                "postprocess_one expects one unbatched VLA output; call unbatch first"
            )

        action_dim = self.describe().action_dim
        if action_dim is not None:
            actions = actions[..., :action_dim]
        return ActionChunk(actions=actions)


__all__ = ["VLAAdapterBase"]
