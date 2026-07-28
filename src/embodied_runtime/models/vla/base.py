"""Shared VLA result semantics layered on the generic model adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.contracts import ActionChunk, ModelPackageError, RawRequest, TensorTree

from ..base import BaseModelAdapter


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

    def postprocess(self, outputs: TensorTree) -> Sequence[ActionChunk]:
        """Temporary compatibility wrapper for the original prototype API."""

        actions = _actions_from(outputs)
        ndim = getattr(actions, "ndim", None)
        if ndim is None:
            raise ModelPackageError("VLA actions must be tensor-like")
        if ndim >= 3:
            samples = self.unbatch(outputs, int(actions.shape[0]))
        else:
            samples = (outputs,)
        return tuple(self.postprocess_one(sample) for sample in samples)


__all__ = ["VLAAdapterBase"]
