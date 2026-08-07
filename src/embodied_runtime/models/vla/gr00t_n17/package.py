"""GR00T policy inspection, spec, and single-forward package construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...errors import ModelPackageError
from ...package import ModelPackage
from ...plans.single_forward import SingleForwardPlan
from ...spec import EntrypointSpec, ModelSpec
from .constants import DEFAULT_ACTION_DIM, DEFAULT_ACTION_HORIZON, DEFAULT_ACTION_KEYS


@dataclass(frozen=True, slots=True)
class Gr00tPackageComponents:
    policy: Any
    embodiment: str
    language_key: str
    action_keys: tuple[str, ...]
    spec: ModelSpec
    package: ModelPackage


class Gr00tPolicyForward:
    """Callable semantic stage; the policy's model is placed by the backend."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def __call__(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        actions, info = self.policy.get_action(dict(observation))
        if not isinstance(actions, Mapping):
            raise ModelPackageError("GR00T policy get_action must return an action mapping")
        if not isinstance(info, Mapping):
            raise ModelPackageError("GR00T policy get_action info must be a mapping")
        return {"actions": dict(actions), "info": dict(info)}


def build_gr00t_package(
    policy: Any,
    *,
    checkpoint: str,
    revision: str | None,
    requested_embodiment: str,
    requested_language_key: str,
    requested_action_dim: Any,
) -> Gr00tPackageComponents:
    try:
        import torch
    except ImportError as error:
        raise ModelPackageError("GR00T N1.7 requires PyTorch and NVIDIA Isaac-GR00T") from error
    model = getattr(policy, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("GR00T policy.model must be torch.nn.Module")
    model.eval()

    embodiment = _embodiment_value(getattr(policy, "embodiment_tag", requested_embodiment))
    language_key = getattr(policy, "language_key", requested_language_key)
    if not isinstance(language_key, str) or not language_key:
        raise ModelPackageError("GR00T policy.language_key must be a non-empty string")

    action_config = _policy_modality_config(policy, "action")
    action_keys = _config_values(action_config, "modality_keys")
    normalized_action_keys = (
        tuple(str(key) for key in action_keys) if action_keys else DEFAULT_ACTION_KEYS
    )
    action_horizon = len(_config_values(action_config, "delta_indices"))
    if action_horizon == 0:
        action_horizon = DEFAULT_ACTION_HORIZON
    action_dim = (
        int(requested_action_dim) if requested_action_dim is not None else DEFAULT_ACTION_DIM
    )
    if action_dim <= 0:
        raise ModelPackageError("GR00T action_dim must be greater than zero")

    spec = ModelSpec(
        model_id=str(checkpoint) if checkpoint else "gr00t-n1.7-injected-policy",
        family="gr00t_n17",
        revision=revision,
        modalities=("vision", "language", "proprioception"),
        action_dim=action_dim,
        action_horizon=action_horizon,
        metadata={
            "framework": "torch",
            "reference_loader": "NVIDIA Isaac-GR00T",
            "execution_plan": "single_forward",
            "embodiment_tag": embodiment,
            "language_key": language_key,
            "action_keys": normalized_action_keys,
            "internal_action_dim": int(
                getattr(getattr(model, "config", None), "max_action_dim", 132)
            ),
        },
    )
    plan = SingleForwardPlan()
    package = ModelPackage(
        spec=spec,
        checkpoint=checkpoint or None,
        plan=plan,
        entrypoints={plan.forward: Gr00tPolicyForward(policy)},
        entrypoint_specs={
            plan.forward: EntrypointSpec(
                plan.forward,
                "Run NVIDIA's official GR00T preprocessing, flow inference, and decoding.",
                batchable=True,
                safe_point_after=True,
                metadata={
                    "output": "named_action_chunk",
                    "internal_generation": "flow_matching",
                },
            )
        },
        metadata={
            "framework": "torch",
            "runtime_module": model,
            "portable": False,
            "reference_runtime": "NVIDIA Isaac-GR00T",
            "embodiment_tag": embodiment,
            "language_key": language_key,
            "action_keys": normalized_action_keys,
        },
    )
    return Gr00tPackageComponents(
        policy=policy,
        embodiment=embodiment,
        language_key=language_key,
        action_keys=normalized_action_keys,
        spec=spec,
        package=package,
    )


def _config_values(config: Any, attribute: str) -> tuple[Any, ...]:
    if config is None:
        return ()
    value = getattr(config, attribute, None)
    if value is None and isinstance(config, Mapping):
        value = config.get(attribute)
    if value is None:
        return ()
    return tuple(value)


def _policy_modality_config(policy: Any, modality: str) -> Any | None:
    configs = getattr(policy, "modality_configs", None)
    if not isinstance(configs, Mapping):
        return None
    return configs.get(modality)


def _embodiment_value(value: Any) -> str:
    resolved = getattr(value, "value", value)
    return str(resolved)
