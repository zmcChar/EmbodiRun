"""OpenVLA-OFT model spec and single-forward package construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...plans.single_forward import SingleForwardPlan
from ...spec import EntrypointSpec, ModelSpec
from .constants import VVLA_SOURCE_COMMIT


@dataclass(frozen=True, slots=True)
class OpenVLAOFTPackageComponents:
    module: Any
    processor: Any
    spec: ModelSpec
    package: ModelPackage


def build_openvla_oft_package(
    module: Any,
    processor: Any,
    *,
    checkpoint: str,
    revision: str | None,
    requested_action_dim: Any,
    requested_action_horizon: Any,
    loader_metadata: dict[str, Any],
) -> OpenVLAOFTPackageComponents:
    action_dim = int(
        requested_action_dim
        if requested_action_dim is not None
        else getattr(module, "action_dim", 7)
    )
    action_horizon = int(
        requested_action_horizon
        if requested_action_horizon is not None
        else getattr(module, "action_horizon", 8)
    )
    if action_dim <= 0 or action_horizon <= 0:
        raise ModelPackageError(
            "OpenVLA-OFT action_dim and action_horizon must be greater than zero"
        )

    module = module.eval()
    spec = ModelSpec(
        model_id="openvla-oft",
        family="openvla_oft_categorical",
        revision=revision,
        modalities=("vision", "language"),
        action_dim=action_dim,
        action_horizon=action_horizon,
        metadata={
            "framework": "torch",
            "generation": "categorical_greedy",
            "execution_plan": "single_forward",
            "checkpoint": checkpoint,
        },
    )
    plan = SingleForwardPlan()
    package = ModelPackage(
        spec=spec,
        checkpoint=checkpoint or None,
        plan=plan,
        entrypoints={plan.forward: module.forward},
        entrypoint_specs={
            plan.forward: EntrypointSpec(
                plan.forward,
                "Run vision, multimodal Llama, and greedy categorical action decoding.",
                batchable=True,
                safe_point_after=True,
                metadata={
                    "generation": "categorical_greedy",
                    "output": "action_chunk",
                },
            )
        },
        metadata={
            "framework": "torch",
            "runtime_module": module,
            "portable": False,
            "generation": "categorical_greedy",
            "loader": loader_metadata,
            "source": {
                "vvla_commit": VVLA_SOURCE_COMMIT,
                "vvla_license": "MIT",
                "rlinf_reference_license": "Apache-2.0",
                "openvla_oft_checkpoint_license": "MIT",
            },
            "limitations": {
                "kv_split": False,
                "rl_logprob_recompute": False,
                "cuda_graph_validated": False,
            },
        },
    )
    return OpenVLAOFTPackageComponents(
        module=module,
        processor=processor,
        spec=spec,
        package=package,
    )
