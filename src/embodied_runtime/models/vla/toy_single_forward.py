"""Tiny single-forward VLA used to prove plan-independent model adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from embodied_runtime.contracts import (
    EntrypointSpec,
    ModelPackage,
    ModelPackageError,
    ModelSpec,
    RawRequest,
    SingleForwardPlan,
)

from ..registry import register_model
from .base import VLAAdapterBase


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - minimal core installation
        raise ModelPackageError(
            "the toy single-forward package requires PyTorch; install the 'torch' extra"
        ) from exc
    return torch


class ToySingleForwardModule:
    """Lazily create a tiny Torch module without importing Torch at discovery."""

    def __new__(cls, action_horizon: int, action_dim: int):
        torch = _torch()

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_horizon = action_horizon
                self.action_dim = action_dim
                self.output_scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
                target = inputs["target"]
                if not isinstance(target, torch.Tensor):
                    target = torch.as_tensor(
                        target,
                        device=self.output_scale.device,
                        dtype=self.output_scale.dtype,
                    )
                if target.ndim != 2:
                    raise ValueError(
                        "toy single-forward target must have shape [batch, action_dim]"
                    )
                if target.shape[-1] == 1:
                    target = target.expand(target.shape[0], self.action_dim)
                if target.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"toy target has action_dim={target.shape[-1]}, expected {self.action_dim}"
                    )
                actions = target[:, None, :].expand(
                    target.shape[0],
                    self.action_horizon,
                    self.action_dim,
                )
                return {"actions": actions * self.output_scale}

        return _Module()


@register_model("toy_single_forward")
class ToySingleForwardAdapter(VLAAdapterBase):
    """ModelAdapter fixture whose package contains one executable entrypoint."""

    def __init__(self, *, action_horizon: int = 2, action_dim: int = 3) -> None:
        if action_horizon <= 0 or action_dim <= 0:
            raise ValueError("action_horizon and action_dim must be greater than zero")
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self._module = None

    def describe(self) -> ModelSpec:
        return ModelSpec(
            model_id="toy-single-forward",
            family="toy_single_forward",
            modalities=("state",),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            metadata={"purpose": "contract-test"},
        )

    def build_package(self, checkpoint: str = "", **options: Any) -> ModelPackage:
        del checkpoint
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"unknown toy single-forward package option(s): {unknown}")
        module = ToySingleForwardModule(self.action_horizon, self.action_dim)
        self._module = module
        plan = SingleForwardPlan()
        return ModelPackage(
            spec=self.describe(),
            plan=plan,
            entrypoints={plan.forward: module.forward},
            entrypoint_specs={
                plan.forward: EntrypointSpec(
                    plan.forward,
                    "Map one batched state target to an action chunk.",
                    batchable=True,
                    safe_point_after=True,
                )
            },
            metadata={
                "framework": "torch",
                "runtime_module": module,
                "portable": False,
                "fixture": True,
            },
        )

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        torch = _torch()
        value = request.observation.get("target", request.observation.get("state"))
        if value is None:
            raise ModelPackageError(
                "toy_single_forward expects observation['target'] or observation['state']"
            )
        target = torch.as_tensor(value, dtype=torch.float32)
        if target.ndim == 0:
            target = target.reshape(1)
        if target.ndim != 1:
            raise ModelPackageError(
                f"toy_single_forward expects one action vector; got shape {tuple(target.shape)}"
            )
        return {"target": target.unsqueeze(0)}


__all__ = ["ToySingleForwardAdapter", "ToySingleForwardModule"]
