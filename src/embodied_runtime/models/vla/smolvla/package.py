"""SmolVLA processor, staged reference module, spec, and package construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...spec import EntrypointSpec, ModelSpec
from .constants import EXPECTED_LEROBOT_VERSION
from .modeling_smolvla import SmolVLAFlowPlan, SmolVLAReferenceModule
from .processing_smolvla import SmolVLAProcessor


@dataclass(frozen=True, slots=True)
class SmolVLAPackageComponents:
    policy: torch.nn.Module
    processor: SmolVLAProcessor
    module: SmolVLAReferenceModule
    spec: ModelSpec
    package: ModelPackage


def build_smolvla_package(
    policy: Any,
    *,
    checkpoint: str,
    revision: str,
    stats_variant: str,
    default_num_steps: int | None,
) -> SmolVLAPackageComponents:
    if not isinstance(policy, torch.nn.Module):
        raise TypeError("SmolVLA policy must be a torch.nn.Module")

    config = policy.config
    rtc_config = getattr(config, "rtc_config", None)
    if rtc_config is not None and bool(getattr(rtc_config, "enabled", False)):
        raise ModelPackageError(
            "RTC-enabled SmolVLA checkpoints are not supported by the staged adapter"
        )

    module = SmolVLAReferenceModule(policy)
    processor = SmolVLAProcessor(policy)
    action_dim = module.native_action_dim
    spec = ModelSpec(
        model_id=str(checkpoint) if checkpoint else "smolvla-injected-policy",
        family="smolvla_flow",
        revision=revision,
        modalities=("vision", "language", "proprioception"),
        action_dim=action_dim,
        action_horizon=int(config.chunk_size),
        metadata={
            "framework": "torch",
            "reference_loader": f"lerobot=={EXPECTED_LEROBOT_VERSION}",
            "image_features": tuple(config.image_features),
            "internal_action_dim": int(config.max_action_dim),
            "stats_variant": stats_variant,
        },
    )
    configured_num_steps = (
        int(config.num_steps) if default_num_steps is None else int(default_num_steps)
    )
    if configured_num_steps <= 0:
        raise ValueError("default_num_steps must be greater than zero")
    plan = SmolVLAFlowPlan(default_num_steps=configured_num_steps)
    package = ModelPackage(
        spec=spec,
        checkpoint=checkpoint or None,
        plan=plan,
        entrypoints={
            plan.encode: module.encode_prefix,
            plan.initialize: module.init_state,
            plan.step: module.denoise_step,
            plan.finalize: module.finalize,
        },
        entrypoint_specs={
            plan.encode: EntrypointSpec(
                plan.encode,
                "Normalize observations and encode SmolVLA prefix KV.",
                batchable=True,
                safe_point_after=True,
            ),
            plan.initialize: EntrypointSpec(
                plan.initialize,
                "Create the FP32 [B, horizon, 32] flow state.",
                batchable=True,
                safe_point_after=True,
            ),
            plan.step: EntrypointSpec(
                plan.step,
                "Return one SmolVLA action velocity field.",
                batchable=True,
                safe_point_after=True,
                metadata={"output": "velocity", "state_update": "engine_owned_euler"},
            ),
            plan.finalize: EntrypointSpec(
                plan.finalize,
                "Crop action padding and apply checkpoint unnormalization.",
                batchable=True,
                safe_point_after=True,
            ),
        },
        metadata={
            "framework": "torch",
            "runtime_module": module,
            "step_output": "velocity",
            "state_update": "state + dt * velocity",
            "portable": False,
            "source": {
                "lerobot_version": EXPECTED_LEROBOT_VERSION,
                "lerobot_license": "Apache-2.0",
                "smolvla_checkpoint_revision": revision,
            },
            "action_contract": {
                "native_dim": action_dim,
                "internal_padded_dim": int(config.max_action_dim),
                "numeric_fusion_with_pi05": False,
            },
        },
    )
    return SmolVLAPackageComponents(
        policy=policy,
        processor=processor,
        module=module,
        spec=spec,
        package=package,
    )
