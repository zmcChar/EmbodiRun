"""Dependency-trimmed real-world evaluator from StreamVLN commit ``e48f6ff7``."""

from __future__ import annotations

import importlib
import re
import time
from collections.abc import Mapping
from typing import Any

from .history import compact_aligned_history
from .preprocessing import (
    BASE_PROMPT,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_MEMORY_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    generation_sources,
    move_payload_to_device,
    preprocess_frame,
    preprocess_qwen,
)

_ACTION_PATTERN = re.compile(r"STOP|↑|←|→")
_ACTION_IDS = {"STOP": 0, "↑": 1, "←": 2, "→": 3}


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

    ``step_id`` remains caller-owned: only the first call of each service cycle
    generates actions; later calls extend the checkpoint's visual timeline.
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
            {"from": "human", "value": BASE_PROMPT},
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
        """Retain bounded, globally spaced history for the next frame block."""

        if not self.frame_ids:
            self._history_count = 0
            return
        compacted = compact_aligned_history(
            self.frame_ids,
            (self.rgb_list, self.depth_list, self.pose_list, self.intrinsic_list),
            next_step_id=next_step_id,
            num_history=self.num_history,
        )
        self.frame_ids = compacted.frame_ids
        (
            self.rgb_list,
            self.depth_list,
            self.pose_list,
            self.intrinsic_list,
        ) = compacted.aligned
        self._history_count = compacted.count

    @staticmethod
    def parse_actions(output: object) -> list[int]:
        """Extract official STOP/arrow tokens in textual order."""

        matches = _ACTION_PATTERN.findall(str(output))
        return [_ACTION_IDS[match] for match in matches]

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
        frame = preprocess_frame(
            rgb,
            run_model=run_model,
            last_image=self.last_image,
            intrinsic_matrix=self.intrinsic_matrix,
            torch_module=self._torch,
            numpy_module=self._np,
            image_module=self._image,
            image_processor=self.image_processor,
        )
        if not run_model and frame.rgb is None:
            raise StreamVLNEvaluatorError("the first evaluator step after reset must run the model")
        if run_model:
            self.last_image = frame.rgb

        self.time_ids.append(self.step_id)
        self.rgb_list.append(frame.rgb)
        self.depth_list.append(frame.depth)
        self.pose_list.append(frame.pose)
        self.intrinsic_list.append(frame.intrinsic)
        self.frame_ids.append(self.step_id)

        if not run_model:
            if self.use_memory_tokens and (self.step_id + 1) % self.num_frames == 0:
                self._compact_history_for_rollover(self.step_id + 1)
                self.model.reset_for_env(environment_id)
                self.output_ids = None
                self.past_key_values = None
                self.time_ids.clear()
            return None, 0.0, None

        sources, add_system = generation_sources(
            self.conversation,
            instruction=instruction,
            step_id=self.step_id,
            has_output_ids=self.output_ids is not None,
        )
        input_ids = preprocess_qwen(
            [sources],
            tokenizer=self.tokenizer,
            torch_module=self._torch,
            add_system=add_system,
        )
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

        payload = move_payload_to_device(
            {
                "images": self._torch.stack(images).unsqueeze(0),
                "depths": self._torch.stack(depths).unsqueeze(0),
                "poses": self._torch.stack(poses).unsqueeze(0),
                "intrinsics": self._torch.stack(intrinsics).unsqueeze(0),
                "inputs": input_ids,
                "env_id": environment_id,
                "time_ids": [self.time_ids],
            },
            torch_module=self._torch,
            device=self.device,
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
