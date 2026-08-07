"""Small Torch flow model used to exercise the model/runtime/backend boundary.

This is deliberately a *contract fixture*, not a benchmark.  It implements the
same four-stage recipe as pi0.5 while remaining tiny enough to run in CPU tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ModelPackageError
from ..package import ModelPackage
from ..plans.iterative_flow import IterativeFlowPlan
from ..registry import register_model
from ..request import RawRequest
from ..spec import EntrypointSpec, ModelSpec
from .base import VLAAdapterBase


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ModelPackageError(
            "the toy flow package requires PyTorch; install the 'torch' extra"
        ) from exc
    return torch


class ToyFlowModule:
    """Lazily constructs a tiny ``torch.nn.Module`` implementation.

    ``ToyFlowModule`` itself is a factory so importing the model registry stays
    possible in installations without PyTorch.  Calling it returns the actual
    module instance.
    """

    def __new__(cls, action_horizon: int, action_dim: int):
        torch = _torch()

        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.action_horizon = action_horizon
                self.action_dim = action_dim
                # A parameter makes device/dtype movement observable to a generic
                # backend without making the fixture computationally expensive.
                self.velocity_scale = torch.nn.Parameter(torch.ones(()))

            def encode_prefix(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
                target = inputs["target"]
                if not isinstance(target, torch.Tensor):
                    target = torch.as_tensor(target, dtype=self.velocity_scale.dtype)
                if target.ndim == 1:
                    target = target[:, None]
                if target.shape[-1] == 1:
                    target = target.expand(target.shape[0], self.action_dim)
                if target.shape[-1] != self.action_dim:
                    raise ValueError(
                        f"toy target has action_dim={target.shape[-1]}, expected {self.action_dim}"
                    )
                expanded = target[:, None, :].expand(
                    target.shape[0], self.action_horizon, self.action_dim
                )
                return {"target": expanded}

            def init_state(self, inputs: Mapping[str, Any]):
                noise = inputs.get("noise")
                if noise is not None:
                    return torch.as_tensor(
                        noise,
                        device=self.velocity_scale.device,
                        dtype=self.velocity_scale.dtype,
                    )
                batch_size = int(inputs["batch_size"])
                generator = inputs.get("generator")
                if generator is None and inputs.get("seed") is not None:
                    generator = torch.Generator(device=self.velocity_scale.device)
                    generator.manual_seed(int(inputs["seed"]))
                return torch.randn(
                    batch_size,
                    self.action_horizon,
                    self.action_dim,
                    device=self.velocity_scale.device,
                    dtype=self.velocity_scale.dtype,
                    generator=generator,
                )

            def denoise_step(self, inputs: Mapping[str, Any]):
                _ = inputs.get("time")  # the fixture's velocity field is time invariant
                state = inputs["state"]
                target = inputs["prefix"]["target"]
                # The pi0.5 recipe has negative dt.  Returning x-target therefore
                # moves x toward target when the engine applies x += dt * velocity.
                return (state - target) * self.velocity_scale

            @staticmethod
            def finalize(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
                return {"actions": inputs["state"]}

        return _Module()


@register_model("toy_flow")
class ToyFlowAdapter(VLAAdapterBase):
    """ModelAdapter for the lightweight staged-flow fixture."""

    def __init__(self, *, action_horizon: int = 3, action_dim: int = 2) -> None:
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self._module = None

    def describe(self) -> ModelSpec:
        return ModelSpec(
            model_id="toy-flow",
            family="toy_flow",
            modalities=("state",),
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            metadata={"purpose": "contract-test"},
        )

    def build_package(self, checkpoint: str = "", **options: Any) -> ModelPackage:
        del checkpoint, options
        module = ToyFlowModule(self.action_horizon, self.action_dim)
        self._module = module
        plan = IterativeFlowPlan(default_num_steps=4)
        specs = {
            plan.encode: EntrypointSpec(
                plan.encode,
                "Encode a target action as the flow condition.",
                safe_point_after=True,
            ),
            plan.initialize: EntrypointSpec(
                plan.initialize,
                "Create or accept the initial flow state.",
                safe_point_after=True,
            ),
            plan.step: EntrypointSpec(
                plan.step,
                "Return velocity; the engine owns the Euler state update.",
                safe_point_after=True,
                metadata={"output": "velocity"},
            ),
            plan.finalize: EntrypointSpec(
                plan.finalize,
                "Expose the integrated state as an action chunk.",
                safe_point_after=True,
            ),
        }
        return ModelPackage(
            spec=self.describe(),
            entrypoints={
                plan.encode: module.encode_prefix,
                plan.initialize: module.init_state,
                plan.step: module.denoise_step,
                plan.finalize: module.finalize,
            },
            plan=plan,
            entrypoint_specs=specs,
            metadata={
                "framework": "torch",
                "runtime_module": module,
                "step_output": "velocity",
                "portable": False,
                "fixture": True,
            },
        )

    def preprocess_one(self, request: RawRequest) -> Mapping[str, Any]:
        torch = _torch()
        observation = request.observation
        value = observation.get("target", observation.get("state"))
        if value is None:
            raise ModelPackageError(
                "toy_flow expects observation['target'] or observation['state']"
            )
        target = torch.as_tensor(value, dtype=torch.float32)
        if target.ndim == 0:
            target = target.reshape(1)
        if target.ndim != 1:
            raise ModelPackageError(
                f"toy_flow expects one action vector; got shape {tuple(target.shape)}"
            )
        return {"target": target.unsqueeze(0)}


__all__ = ["ToyFlowAdapter", "ToyFlowModule"]
