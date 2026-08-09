"""Pinned ActiveVLN protocol constants and checkpoint guards."""

from __future__ import annotations

from typing import Any

from embodied_runtime.policies.navigation.errors import NavigationPolicyError

SYSTEM_PROMPT_R2R = (
    "You are a helpful assistant. "
    "Your goal is to follow the given instruction to reach a specified destination. \n"
    "At each step, you receive a first-person image (starting view if first step (step 1), or "
    "post-action view otherwise). "
    "Your task is to select choose one action from: move forward 25cm, move forward 50cm, "
    "move forward 75cm, turn left 15 degrees, turn left 30 degrees, turn left 45 degrees, "
    "turn right 15 degrees, turn right 30 degrees, turn right 45 degrees, or stop. \n"
    "The instruction will be provided with each observation. You can take multiple actions at each turn. "
)
USER_SUFFIX = (
    "Instruction: {}Decide your next action. "
    "You can take up to 3 actions at a time, separated by ','. "
)
MAX_HISTORY_IMAGES = 200


def validate_checkpoint_model(model: Any) -> None:
    """Fail closed when a local checkpoint is not the pinned Qwen2.5-VL shape."""

    config = model.config
    text = getattr(config, "text_config", config)
    if config.model_type != "qwen2_5_vl":
        raise NavigationPolicyError(
            f"ActiveVLN needs model_type='qwen2_5_vl', got {config.model_type!r}"
        )
    architectures = getattr(config, "architectures", ())
    if "Qwen2_5_VLForConditionalGeneration" not in architectures:
        raise NavigationPolicyError(f"unexpected ActiveVLN architectures: {architectures!r}")
    sections = list(text.rope_scaling["mrope_section"])
    if sections != [16, 24, 24]:
        raise NavigationPolicyError(f"unexpected ActiveVLN mRoPE sections: {sections!r}")


__all__ = ["MAX_HISTORY_IMAGES", "SYSTEM_PROMPT_R2R", "USER_SUFFIX", "validate_checkpoint_model"]
