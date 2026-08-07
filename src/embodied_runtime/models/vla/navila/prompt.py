"""Pinned eight-image conversation prompt used by the released NaVILA checkpoint."""

from __future__ import annotations

from .config import REFERENCE_SOURCE_COMMIT

NAVILA_STOP_STRING = "<|eot_id|>"
PROMPT_SOURCE_ANCHORS = (
    f"navila_trainer.py:175-180@{REFERENCE_SOURCE_COMMIT}",
    f"conversation.py:85-94,453-464@{REFERENCE_SOURCE_COMMIT}",
)

_SYSTEM = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "You are a helpful language and vision assistant. You are able to understand the visual "
    "content that the user provides, and assist the user with a variety of tasks using natural "
    "language."
)
_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def build_navila_prompt(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    if len(instruction) > 4096:
        raise ValueError("instruction exceeds 4096 characters")
    question = (
        "Imagine you are a robot programmed for navigation tasks. You have been given a video "
        "of historical observations "
        + ("<image>\n" * 7)
        + ', and current observation <image>\n. Your assigned task is: "'
        + instruction
        + '" Analyze this series of images to decide your next action, which could be turning '
        "left or right by a specific degree, moving forward a certain distance, or stop if the "
        "task is completed."
    )
    return _SYSTEM + NAVILA_STOP_STRING + _USER + question + NAVILA_STOP_STRING + _ASSISTANT


__all__ = [
    "NAVILA_STOP_STRING",
    "PROMPT_SOURCE_ANCHORS",
    "build_navila_prompt",
]
