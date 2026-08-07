"""Prompt, image, and tensor preprocessing for StreamVLN evaluation."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_MEMORY_TOKEN = "<memory>"
IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -300

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
    "+ '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
BASE_PROMPT = (
    "<video>\nYou are an autonomous navigation assistant. Your task is to "
    "<instruction>. Devise an action sequence to follow the instruction using "
    "the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
    "MOVE FORWARD (↑) by 25 centimeters, or STOP."
)


@dataclass(frozen=True, slots=True)
class FrameInputs:
    """One visual frame and its shape-compatible model placeholders."""

    rgb: Any
    depth: Any
    pose: Any
    intrinsic: Any


def preprocess_qwen(
    sources: Sequence[Sequence[Mapping[str, str]]],
    *,
    tokenizer: Any,
    torch_module: Any,
    add_system: bool,
    system_message: str = "You are a helpful assistant.",
) -> Any:
    """Encode official Qwen conversations and replace visual token IDs."""

    roles = {"human": "user", "gpt": "assistant"}
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    memory_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_MEMORY_TOKEN)
    tokenizer.chat_template = CHAT_TEMPLATE

    encoded_conversations: list[list[int]] = []
    for original_source in sources:
        source = [dict(message) for message in original_source]
        visual_prompt = f"you can see {DEFAULT_IMAGE_TOKEN}"
        if source[0]["value"]:
            source[0]["value"] += f" {visual_prompt}."
        else:
            source[0]["value"] = f"{visual_prompt}."
        if roles.get(source[0]["from"], source[0]["from"]) != roles["human"]:
            source = source[1:]

        input_ids: list[int] = []
        if add_system:
            input_ids.extend(
                tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
            )
        for conversation in source:
            role = conversation.get("role", conversation.get("from", ""))
            content = conversation.get("content", conversation.get("value", ""))
            input_ids.extend(
                tokenizer.apply_chat_template([{"role": roles.get(role, role), "content": content}])
            )

        encoded_conversations.append(
            [
                IMAGE_TOKEN_INDEX
                if token_id == image_token_id
                else MEMORY_TOKEN_INDEX
                if token_id == memory_token_id
                else token_id
                for token_id in input_ids
            ]
        )
    return torch_module.tensor(encoded_conversations, dtype=torch_module.long)


def preprocess_rgb_image(rgb: Any, *, image_module: Any, image_processor: Any) -> Any:
    """Convert an RGB array to one detached vision-tower input tensor."""

    image = image_module.fromarray(rgb).convert("RGB")
    processed = image_processor.preprocess(images=image, return_tensors="pt")
    return copy.deepcopy(processed["pixel_values"][0])


def preprocess_frame(
    rgb: Any,
    *,
    run_model: bool,
    last_image: Any,
    intrinsic_matrix: Any,
    torch_module: Any,
    numpy_module: Any,
    image_module: Any,
    image_processor: Any,
) -> FrameInputs:
    """Validate one frame and build the official model's placeholder features."""

    try:
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError("rgb must be an HxWx3 array-like image") from error
    if height < 1 or width < 1:
        raise ValueError("rgb dimensions must be greater than zero")

    pose = numpy_module.eye(4)
    depth = numpy_module.zeros((height, width, 1))
    intrinsic = torch_module.from_numpy(intrinsic_matrix).float()
    image_tensor = (
        preprocess_rgb_image(
            rgb,
            image_module=image_module,
            image_processor=image_processor,
        )
        if run_model
        else last_image
    )
    return FrameInputs(
        rgb=image_tensor,
        depth=torch_module.from_numpy(depth).float(),
        pose=torch_module.from_numpy(pose),
        intrinsic=intrinsic,
    )


def move_payload_to_device(
    payload: dict[str, Any],
    *,
    torch_module: Any,
    device: Any,
) -> dict[str, Any]:
    """Move tensor values and tensor lists to the evaluator device in place."""

    tensor_type = getattr(torch_module, "Tensor", ())
    for key, value in payload.items():
        if isinstance(value, tensor_type):
            payload[key] = value.to(device)
        elif isinstance(value, list) and value and isinstance(value[0], tensor_type):
            payload[key] = [element.to(device) for element in value]
    return payload


def generation_sources(
    conversation: Sequence[Mapping[str, str]],
    *,
    instruction: str,
    step_id: int,
    has_output_ids: bool,
) -> tuple[list[dict[str, str]], bool]:
    """Build either an initial navigation prompt or a recurrent continuation."""

    if has_output_ids:
        return ([{"from": "human", "value": ""}, {"from": "gpt", "value": ""}], False)

    sources = [dict(message) for message in copy.deepcopy(conversation)]
    sources[0]["value"] = sources[0]["value"].replace(
        " Where should you go next to stay on track?",
        " Please devise an action sequence to follow the instruction which may include "
        "turning left or right by a certain degree, moving forward by a certain distance "
        "or stopping once the task is complete.",
    )
    if step_id != 0:
        sources[0]["value"] += f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
    sources[0]["value"] = sources[0]["value"].replace(f"{DEFAULT_VIDEO_TOKEN}\n", "")
    sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
    return sources, True


__all__ = [
    "BASE_PROMPT",
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_MEMORY_TOKEN",
    "DEFAULT_VIDEO_TOKEN",
    "IMAGE_TOKEN_INDEX",
    "MEMORY_TOKEN_INDEX",
    "FrameInputs",
    "generation_sources",
    "move_payload_to_device",
    "preprocess_frame",
    "preprocess_qwen",
    "preprocess_rgb_image",
]
