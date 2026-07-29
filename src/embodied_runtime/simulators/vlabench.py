"""Lazy VLABench endpoint used by simulator-first deployment experiments.

The module deliberately keeps LeRobot, VLABench, NumPy, and MuJoCo behind the
runtime boundary.  Importing :mod:`embodied_runtime.simulators` therefore does
not require the dedicated simulation environment.
"""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Mapping
from typing import Any

from embodied_runtime.contracts import RobotAction, RobotObservation

from .base import EpisodeStep, SimulatorCapabilities

VLABENCH_ACTION_DIM = 7


class VLABenchSimulatorEndpoint:
    """Expose one LeRobot VLABench environment through the simulator contract."""

    def __init__(
        self,
        task: str,
        *,
        max_episode_steps: int = 500,
        render_resolution: tuple[int, int] = (480, 480),
        environment_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("VLABench task must be a non-empty string")
        if (
            isinstance(max_episode_steps, bool)
            or not isinstance(max_episode_steps, int)
            or max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer")
        if (
            not isinstance(render_resolution, tuple)
            or len(render_resolution) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in render_resolution
            )
        ):
            raise ValueError("render_resolution must contain two positive integers")
        if environment_factory is not None and not callable(environment_factory):
            raise TypeError("environment_factory must be callable or None")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self.task = task.strip()
        self.max_episode_steps = max_episode_steps
        self.render_resolution = render_resolution
        self._environment_factory = environment_factory or _make_environment
        self._clock = clock
        self._environment: Any | None = None
        self._last_observation: RobotObservation | None = None
        self._instruction: str | None = None
        self._initial_fingerprint: str | None = None
        self._closed = False
        self._step_index = 0
        self._capabilities = SimulatorCapabilities(
            name="vlabench",
            environment_id=self.task,
            embodiment="franka_panda",
            action_space_id="eef_xyz_euler_gripper_v1",
            supports_seed=True,
            supports_observe=True,
            features=frozenset(
                {
                    "rgb_front",
                    "rgb_secondary",
                    "rgb_wrist",
                    "eef_state",
                    "task_instruction",
                    "ground_truth_success",
                }
            ),
            metadata={
                "action_dim": VLABENCH_ACTION_DIM,
                "max_episode_steps": max_episode_steps,
                "render_resolution": render_resolution,
            },
        )

    @property
    def capabilities(self) -> SimulatorCapabilities:
        return self._capabilities

    @property
    def instruction(self) -> str:
        if self._instruction is None:
            raise RuntimeError("VLABench endpoint must be reset before reading its instruction")
        return self._instruction

    @property
    def initial_fingerprint(self) -> str:
        if self._initial_fingerprint is None:
            raise RuntimeError("VLABench endpoint must be reset before reading its fingerprint")
        return self._initial_fingerprint

    @property
    def raw_environment(self) -> Any:
        """Return the loaded environment for integration-only oracle inspection."""

        self._ensure_open()
        if self._environment is None:
            raise RuntimeError("VLABench endpoint must be reset before reading its environment")
        return self._environment

    def reset(
        self,
        task: str | None = None,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> RobotObservation:
        self._ensure_open()
        if task is not None and task.strip() != self.task:
            raise ValueError(f"endpoint is bound to VLABench task {self.task!r}, not {task!r}")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("VLABench seed must be an integer or None")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("VLABench reset options must be a mapping or None")

        self._seed_process(seed)
        environment = self._get_or_create_environment()
        reset_kwargs: dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options:
            reset_kwargs["options"] = dict(options)
        raw_observation, reset_info = _unpack_reset(environment.reset(**reset_kwargs))

        instruction = _extract_instruction(environment, fallback=self.task)
        if hasattr(environment, "task_description"):
            environment.task_description = instruction
        self._instruction = instruction
        self._step_index = 0
        observation = self._wrap_observation(
            raw_observation,
            metadata={
                "task": self.task,
                "instruction": instruction,
                "seed": seed,
                "reset_info": dict(reset_info),
                "step_index": 0,
            },
        )
        self._last_observation = observation
        self._initial_fingerprint = observation_fingerprint(observation.values)
        observation.metadata["initial_fingerprint"] = self._initial_fingerprint
        return observation

    def observe(self) -> RobotObservation:
        self._ensure_open()
        if self._last_observation is None:
            raise RuntimeError("VLABench endpoint must be reset before observe()")
        return self._last_observation

    def step(self, action: RobotAction) -> EpisodeStep:
        self._ensure_open()
        if self._environment is None or self._last_observation is None:
            raise RuntimeError("VLABench endpoint must be reset before step()")
        if not isinstance(action, RobotAction):
            raise TypeError("VLABench action must be a RobotAction")

        action_array = _action_array(action.values)
        raw_observation, reward, terminated, truncated, info = _unpack_step(
            self._environment.step(action_array)
        )
        self._step_index += 1
        success = _success_from_info(info)
        observation = self._wrap_observation(
            raw_observation,
            metadata={
                "task": self.task,
                "instruction": self.instruction,
                "step_index": self._step_index,
            },
        )
        self._last_observation = observation
        return EpisodeStep(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated or self._step_index >= self.max_episode_steps,
            success=success,
            info=info,
        )

    def close(self) -> None:
        if self._closed:
            return
        if self._environment is not None:
            close = getattr(self._environment, "close", None)
            if callable(close):
                close()
        self._environment = None
        self._closed = True

    def _get_or_create_environment(self) -> Any:
        if self._environment is None:
            self._environment = self._environment_factory(
                task=self.task,
                obs_type="pixels_agent_pos",
                render_mode="rgb_array",
                render_resolution=self.render_resolution,
                robot="franka",
                max_episode_steps=self.max_episode_steps,
                action_mode="eef",
            )
        return self._environment

    def _seed_process(self, seed: int | None) -> None:
        if seed is None:
            return
        # VLABench samples layouts through both module-level RNGs before its
        # inner dm-control reset sees a seed.  Seeding both immediately before
        # lazy environment construction makes paired conditions reproducible;
        # the initial observation fingerprint is still checked by the harness.
        random.seed(seed)
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError(
                "VLABench execution requires NumPy in the isolated simulator environment"
            ) from error
        np.random.seed(seed)

    def _wrap_observation(
        self,
        values: Any,
        *,
        metadata: Mapping[str, Any],
    ) -> RobotObservation:
        if not isinstance(values, Mapping):
            raise TypeError("VLABench observation must be a mapping")
        return RobotObservation(
            timestamp_s=float(self._clock()),
            values=dict(values),
            metadata=dict(metadata),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("VLABench endpoint is closed")


def observation_fingerprint(values: Any) -> str:
    """Return a stable content digest for paired-reset validation."""

    digest = hashlib.sha256()
    _update_digest(digest, values)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"=")
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"sequence:{len(value)}:".encode())
        for item in value:
            _update_digest(digest, item)
        return

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    tobytes = getattr(value, "tobytes", None)
    if shape is not None and dtype is not None and callable(tobytes):
        digest.update(f"array:{tuple(shape)}:{dtype}:".encode())
        digest.update(tobytes())
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
        return
    digest.update(f"scalar:{type(value).__name__}:{value!r}".encode())


def _make_environment(**kwargs: Any) -> Any:
    from embodied_runtime.integrations.lerobot.vlabench_camera import (
        make_semantic_vlabench_environment,
    )

    return make_semantic_vlabench_environment(**kwargs)


def _extract_instruction(environment: Any, *, fallback: str) -> str:
    task_object = getattr(getattr(environment, "_env", None), "task", None)
    get_instruction = getattr(task_object, "get_instruction", None)
    if callable(get_instruction):
        instruction = get_instruction()
        if isinstance(instruction, str) and instruction.strip():
            return instruction.strip()
        if isinstance(instruction, (tuple, list)):
            for candidate in instruction:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

    for owner in (task_object, environment):
        for name in ("task_description", "language_instruction", "instruction"):
            candidate = getattr(owner, name, None)
            if isinstance(candidate, str) and candidate.strip() and candidate != fallback:
                return candidate.strip()
        candidates = getattr(owner, "instructions", None)
        if isinstance(candidates, (tuple, list)):
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
    return fallback


def _action_array(values: Any) -> Any:
    raw = values.get("action") if isinstance(values, Mapping) and "action" in values else values
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "VLABench execution requires NumPy in the isolated simulator environment"
        ) from error
    action = np.asarray(raw, dtype=np.float32)
    if action.shape != (VLABENCH_ACTION_DIM,):
        raise ValueError(
            f"VLABench action must have shape ({VLABENCH_ACTION_DIM},), got {action.shape}"
        )
    if not np.isfinite(action).all():
        raise ValueError("VLABench action must contain only finite values")
    return action


def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        if not isinstance(info, Mapping):
            raise TypeError("VLABench reset info must be a mapping")
        return observation, info
    return result, {}


def _unpack_step(
    result: Any,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError("VLABench step must return five values")
    observation, reward, terminated, truncated, info = result
    if not isinstance(info, Mapping):
        raise TypeError("VLABench step info must be a mapping")
    return observation, float(reward), bool(terminated), bool(truncated), dict(info)


def _success_from_info(info: Mapping[str, Any]) -> bool | None:
    value = info.get("is_success", info.get("success"))
    if value is None:
        return None
    return bool(value)


__all__ = [
    "VLABENCH_ACTION_DIM",
    "VLABenchSimulatorEndpoint",
    "observation_fingerprint",
]
