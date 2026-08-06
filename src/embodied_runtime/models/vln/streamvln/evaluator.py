"""Dependency-trimmed evaluator for StreamVLN real-world inference.

This is the RGB/model portion of the official ``VLNEvaluator`` at StreamVLN
commit ``e48f6ff7``.  The upstream module imports simulation, ROS-adjacent
geometry, depth filtering, and distributed helpers at module import time even
though its real-world ``step`` method does not use them.  Keeping the evaluator
here lets the Go2 endpoint share a Python 3.10 / PyTorch 2.5 / Transformers
4.51 environment without installing that unused stack.

Heavy dependencies are resolved only when an evaluator is constructed.  Merely
importing :mod:`embodied_runtime.models.vln.streamvln` stays dependency-free.
"""

from __future__ import annotations

import copy
import importlib
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_MEMORY_TOKEN = "<memory>"
IMAGE_TOKEN_INDEX = -200
MEMORY_TOKEN_INDEX = -300

_ACTION_PATTERN = re.compile(r"STOP|↑|←|→")
_ACTION_IDS = {"STOP": 0, "↑": 1, "←": 2, "→": 3}
_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
    "+ '<|im_end|>' + '\\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
_BASE_PROMPT = (
    "<video>\nYou are an autonomous navigation assistant. Your task is to "
    "<instruction>. Devise an action sequence to follow the instruction using "
    "the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, "
    "MOVE FORWARD (↑) by 25 centimeters, or STOP."
)


class StreamVLNEvaluatorError(RuntimeError):
    """The dependency-trimmed evaluator could not execute a model step."""


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_dependency(name: str, purpose: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:  # pragma: no cover - deployment-only failure
        raise StreamVLNEvaluatorError(
            f"StreamVLN {purpose} requires the {name!r} package"
        ) from error


class StreamVLNEvaluator:
    """Stateful recurrent evaluator used by the official Go2 checkpoint.

    ``step_id`` deliberately remains caller-owned, matching upstream: a
    four-step service cycle calls :meth:`step` four times and increments it
    after each call.  Only the first call performs generation; the remaining
    calls extend the visual timeline used by the checkpoint's memory cadence.
    """

    def __init__(
        self,
        sensor_config: Mapping[str, Any],
        *,
        model: Any,
        tokenizer: Any,
        device: Any,
        num_frames: int = 32,
        num_future_steps: int = 4,
        num_history: int = 8,
        max_new_tokens: int = 10_000,
        environment_id: int = 0,
        torch_module: Any | None = None,
        numpy_module: Any | None = None,
        pillow_image_module: Any | None = None,
    ) -> None:
        if "camera_intrinsic" not in sensor_config:
            raise ValueError("sensor_config must contain camera_intrinsic")
        self.num_frames = _positive_int(num_frames, "num_frames")
        self.num_future_steps = _positive_int(num_future_steps, "num_future_steps")
        self.num_history = _positive_int(num_history, "num_history")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        if isinstance(environment_id, bool) or not isinstance(environment_id, int):
            raise ValueError("environment_id must be an integer")  # noqa: TRY004

        self._torch = torch_module or _load_dependency("torch", "inference")
        self._np = numpy_module or _load_dependency("numpy", "inference")
        self._image = pillow_image_module or _load_dependency("PIL.Image", "image decoding")
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.environment_id = environment_id
        self.intrinsic_matrix = self._np.asarray(sensor_config["camera_intrinsic"])
        self.image_processor = model.get_vision_tower().image_processor

        # These tokens already exist in the published checkpoint.  Calling
        # add_tokens is nevertheless part of the upstream initialization and
        # keeps locally converted tokenizers compatible.
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_MEMORY_TOKEN], special_tokens=True)
        self.conversation = (
            {"from": "human", "value": _BASE_PROMPT},
            {"from": "gpt", "value": ""},
        )

        self.use_memory_tokens = True
        self.rgb_list: list[Any] = []
        self.depth_list: list[Any] = []
        self.pose_list: list[Any] = []
        self.intrinsic_list: list[Any] = []
        self.frame_ids: list[int] = []
        self._history_count = 0
        self.time_ids: list[int] = []
        self.output_ids: Any | None = None
        self.past_key_values: Any | None = None
        self.step_id = 0
        self.last_image: Any | None = None

    def reset_memory(self) -> None:
        """Reset both Python-side history and the checkpoint's environment cache."""

        self.rgb_list.clear()
        self.depth_list.clear()
        self.pose_list.clear()
        self.intrinsic_list.clear()
        self.frame_ids.clear()
        self._history_count = 0
        self.time_ids.clear()
        self.output_ids = None
        self.past_key_values = None
        self.step_id = 0
        self.last_image = None
        self.model.reset_for_env(self.environment_id)

    def _compact_history_for_rollover(self, next_step_id: int) -> None:
        """Retain a bounded, evenly spaced history for the next 32-frame block.

        Upstream keeps every virtual frame forever and selects uniformly spaced
        history only when its memory window rolls over.  Selecting that same
        effective history before clearing the completed window bounds all four
        aligned lists to ``num_history + num_frames`` entries.  On very long
        episodes, already compacted candidates are matched to the nearest
        global target rather than allowing storage to grow without bound.
        """

        if not self.frame_ids:
            self._history_count = 0
            return
        stride = max(1, next_step_id // self.num_history)
        targets = tuple(range(0, next_step_id, stride))[: self.num_history]
        available = set(range(len(self.frame_ids)))
        selected: list[int] = []
        for target in targets:
            if not available:
                break
            index = min(
                available,
                key=lambda candidate: (
                    abs(self.frame_ids[candidate] - target),
                    self.frame_ids[candidate] > target,
                    self.frame_ids[candidate],
                ),
            )
            selected.append(index)
            available.remove(index)
        selected.sort(key=self.frame_ids.__getitem__)

        self.rgb_list = [self.rgb_list[index] for index in selected]
        self.depth_list = [self.depth_list[index] for index in selected]
        self.pose_list = [self.pose_list[index] for index in selected]
        self.intrinsic_list = [self.intrinsic_list[index] for index in selected]
        self.frame_ids = [self.frame_ids[index] for index in selected]
        self._history_count = len(selected)

    @staticmethod
    def parse_actions(output: object) -> list[int]:
        """Extract official STOP/arrow tokens in textual order."""

        matches = _ACTION_PATTERN.findall(str(output))
        return [_ACTION_IDS[match] for match in matches]

    def _preprocess_qwen(
        self,
        sources: Sequence[Sequence[Mapping[str, str]]],
        *,
        add_system: bool,
        system_message: str = "You are a helpful assistant.",
    ) -> Any:
        roles = {"human": "user", "gpt": "assistant"}
        image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        memory_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_MEMORY_TOKEN)
        self.tokenizer.chat_template = _CHAT_TEMPLATE

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
                    self.tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_message}]
                    )
                )
            for conversation in source:
                role = conversation.get("role", conversation.get("from", ""))
                content = conversation.get("content", conversation.get("value", ""))
                input_ids.extend(
                    self.tokenizer.apply_chat_template(
                        [{"role": roles.get(role, role), "content": content}]
                    )
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
        return self._torch.tensor(encoded_conversations, dtype=self._torch.long)

    def _to_device(self, payload: dict[str, Any]) -> dict[str, Any]:
        tensor_type = getattr(self._torch, "Tensor", ())
        for key, value in payload.items():
            if isinstance(value, tensor_type):
                payload[key] = value.to(self.device)
            elif isinstance(value, list) and value and isinstance(value[0], tensor_type):
                payload[key] = [element.to(self.device) for element in value]
        return payload

    def _generation_sources(self, instruction: str) -> tuple[list[dict[str, str]], bool]:
        if self.output_ids is not None:
            return ([{"from": "human", "value": ""}, {"from": "gpt", "value": ""}], False)

        sources = [dict(message) for message in copy.deepcopy(self.conversation)]
        sources[0]["value"] = sources[0]["value"].replace(
            " Where should you go next to stay on track?",
            " Please devise an action sequence to follow the instruction which may include "
            "turning left or right by a certain degree, moving forward by a certain distance "
            "or stopping once the task is complete.",
        )
        if self.step_id != 0:
            sources[0]["value"] += f" You have visited these areas {DEFAULT_MEMORY_TOKEN}."
        sources[0]["value"] = sources[0]["value"].replace(f"{DEFAULT_VIDEO_TOKEN}\n", "")
        sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
        return sources, True

    def step(
        self,
        environment_id: int,
        rgb: Any,
        instruction: str = "",
        *,
        run_model: bool = False,
    ) -> tuple[list[int] | None, float, str | None]:
        """Advance one official StreamVLN virtual frame.

        The real-world checkpoint ignores depth and pose features today, but
        its model signature still requires shape-compatible placeholders.
        """

        if environment_id != self.environment_id:
            raise ValueError(
                f"this evaluator owns environment {self.environment_id}, got {environment_id}"
            )
        try:
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise ValueError("rgb must be an HxWx3 array-like image") from error
        if height < 1 or width < 1:
            raise ValueError("rgb dimensions must be greater than zero")

        camera_pose = self._np.eye(4)
        depth = self._np.zeros((height, width, 1))
        intrinsic = self._torch.from_numpy(self.intrinsic_matrix).float()

        if run_model:
            image = self._image.fromarray(rgb).convert("RGB")
            processed = self.image_processor.preprocess(images=image, return_tensors="pt")
            image_tensor = processed["pixel_values"][0]
            self.last_image = copy.deepcopy(image_tensor)
        else:
            if self.last_image is None:
                raise StreamVLNEvaluatorError(
                    "the first evaluator step after reset must run the model"
                )
            image_tensor = self.last_image

        self.time_ids.append(self.step_id)
        self.rgb_list.append(image_tensor)
        self.depth_list.append(self._torch.from_numpy(depth).float())
        self.pose_list.append(self._torch.from_numpy(camera_pose))
        self.intrinsic_list.append(intrinsic)
        self.frame_ids.append(self.step_id)

        if not run_model:
            if self.use_memory_tokens and (self.step_id + 1) % self.num_frames == 0:
                self._compact_history_for_rollover(self.step_id + 1)
                self.model.reset_for_env(environment_id)
                self.output_ids = None
                self.past_key_values = None
                self.time_ids.clear()
            return None, 0.0, None

        sources, add_system = self._generation_sources(instruction)
        input_ids = self._preprocess_qwen([sources], add_system=add_system)
        if self.output_ids is not None:
            input_ids = self._torch.cat(
                (self.output_ids, input_ids.to(self.output_ids.device)), dim=1
            )

        images = self.rgb_list[-1:]
        depths = self.depth_list[-1:]
        poses = self.pose_list[-1:]
        intrinsics = self.intrinsic_list[-1:]
        if self.use_memory_tokens and self.step_id and self.step_id % self.num_frames == 0:
            history = slice(0, self._history_count)
            images = self.rgb_list[history] + images
            depths = self.depth_list[history] + depths
            poses = self.pose_list[history] + poses
            intrinsics = self.intrinsic_list[history] + intrinsics

        payload = self._to_device(
            {
                "images": self._torch.stack(images).unsqueeze(0),
                "depths": self._torch.stack(depths).unsqueeze(0),
                "poses": self._torch.stack(poses).unsqueeze(0),
                "intrinsics": self._torch.stack(intrinsics).unsqueeze(0),
                "inputs": input_ids,
                "env_id": environment_id,
                "time_ids": [self.time_ids],
            }
        )
        for key in ("images", "depths", "poses", "intrinsics"):
            payload[key] = payload[key].to(self._torch.bfloat16)

        started_at = time.monotonic()
        outputs = self.model.generate(
            **payload,
            task_ids=[0],
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self.past_key_values,
        )
        generate_time = time.monotonic() - started_at
        self.output_ids = outputs.sequences
        decoded = self.tokenizer.batch_decode(self.output_ids, skip_special_tokens=False)[0].strip()
        self.past_key_values = outputs.past_key_values
        actions = self.parse_actions(decoded)
        if not actions:
            actions = [0]
        return actions, generate_time, decoded


__all__ = [
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_MEMORY_TOKEN",
    "DEFAULT_VIDEO_TOKEN",
    "IMAGE_TOKEN_INDEX",
    "MEMORY_TOKEN_INDEX",
    "StreamVLNEvaluator",
    "StreamVLNEvaluatorError",
]
