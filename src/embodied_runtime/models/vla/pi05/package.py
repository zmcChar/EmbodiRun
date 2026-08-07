"""pi0.5 processor, reference stages, spec, and package construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...plans.iterative_flow import IterativeFlowPlan
from ...spec import EntrypointSpec, ModelSpec
from .modeling_pi05 import Pi05ReferenceModule
from .processing_pi05 import Pi05Processor

VVLA_SOURCE_COMMIT = "80b5cf48c8710c69ed97200903562e9787efe105"


@dataclass(frozen=True, slots=True)
class Pi05PackageComponents:
    policy: Any
    processor: Pi05Processor
    module: Pi05ReferenceModule
    spec: ModelSpec
    package: ModelPackage


def build_pi05_package(
    policy: Any,
    *,
    checkpoint: str,
    revision: str | None,
) -> Pi05PackageComponents:
    config = policy.config
    rtc_config = getattr(config, "rtc_config", None)
    if rtc_config is not None and bool(getattr(rtc_config, "enabled", False)):
        raise ModelPackageError(
            "RTC-enabled pi0.5 checkpoints are not supported by the staged "
            "prototype because RTC changes the denoise-loop semantics"
        )

    processor = Pi05Processor(policy)
    module = Pi05ReferenceModule(policy)
    output_features = getattr(config, "output_features", None) or {}
    action_feature = output_features.get("action")
    action_dim = (
        int(action_feature.shape[0]) if action_feature is not None else int(config.max_action_dim)
    )
    spec = ModelSpec(
        model_id=str(checkpoint) if checkpoint else "pi05-injected-policy",
        family="pi05_flow",
        revision=revision,
        modalities=("vision", "language", "proprioception"),
        action_dim=action_dim,
        action_horizon=int(config.chunk_size),
        metadata={
            "framework": "torch",
            "reference_loader": "lerobot==0.5.1",
            "image_features": tuple(config.image_features),
            "image_resolution": tuple(config.image_resolution),
            "internal_action_dim": int(config.max_action_dim),
        },
    )
    plan = IterativeFlowPlan(default_num_steps=int(config.num_inference_steps))
    specs = {
        plan.encode: EntrypointSpec(
            plan.encode,
            "Encode images and tokens into per-layer prefix K/V.",
            batchable=True,
            safe_point_after=True,
        ),
        plan.initialize: EntrypointSpec(
            plan.initialize,
            "Create initial action noise; explicit noise overrides generator/seed.",
            batchable=True,
            safe_point_after=True,
        ),
        plan.step: EntrypointSpec(
            plan.step,
            "Return the pi0.5 velocity field for one flow time.",
            batchable=True,
            safe_point_after=True,
            metadata={"output": "velocity", "state_update": "engine_owned_euler"},
        ),
        plan.finalize: EntrypointSpec(
            plan.finalize,
            "Expose the integrated state as actions.",
            batchable=True,
            safe_point_after=True,
        ),
    }
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
        entrypoint_specs=specs,
        metadata={
            "framework": "torch",
            "runtime_module": module,
            "step_output": "velocity",
            "state_update": "state + dt * velocity",
            "portable": False,
            "source": {
                "vvla_commit": VVLA_SOURCE_COMMIT,
                "vvla_license": "MIT",
                "lerobot_version": "0.5.1",
                "lerobot_license": "Apache-2.0",
                "openpi_origin": "Physical-Intelligence/openpi",
            },
        },
    )
    return Pi05PackageComponents(
        policy=policy,
        processor=processor,
        module=module,
        spec=spec,
        package=package,
    )
