"""Native LeRobot adapter for the VLABench SmolVLA checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from .inspection import (
    feature_shapes,
    one_cpu_action,
    parameter_count,
    policy_action_queue,
    processor_stat_shapes,
)
from .native import load_native_bindings
from .values import (
    VLABENCH_CAMERA_RENAME_MAP,
    LeRobotBindings,
    LeRobotRunnerError,
    SmolVLAAction,
    SmolVLARuntimeFacts,
)


class VLABenchSmolVLARunner:
    """Run ``lerobot/smolvla_vlabench`` through LeRobot's native pipelines."""

    def __init__(
        self,
        checkpoint: str = "lerobot/smolvla_vlabench",
        *,
        task: str = "select_fruit",
        device: str = "cuda",
        revision: str | None = None,
        backbone_path: str | None = None,
        local_files_only: bool = False,
        bindings: LeRobotBindings | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not checkpoint.strip():
            raise ValueError("checkpoint must not be empty")
        if not task.strip():
            raise ValueError("task must not be empty")
        if not device.strip():
            raise ValueError("device must not be empty")
        if backbone_path is not None and not backbone_path.strip():
            raise ValueError("backbone_path must be non-empty when provided")

        self.checkpoint = checkpoint
        self.task = task
        self.device = device
        self.revision = revision
        self.backbone_path = None if backbone_path is None else backbone_path.strip()
        self.local_files_only = local_files_only
        self._bindings = bindings
        self._clock = clock
        self._config: Any | None = None
        self._environment_config: Any | None = None
        self._policy: Any | None = None
        self._preprocessor: Any | None = None
        self._postprocessor: Any | None = None
        self._environment_preprocessor: Any | None = None
        self._environment_postprocessor: Any | None = None
        self._facts: SmolVLARuntimeFacts | None = None
        self._active_prompt: str | None = None

    @property
    def loaded(self) -> bool:
        return self._policy is not None

    @property
    def facts(self) -> SmolVLARuntimeFacts:
        if self._facts is None:
            raise LeRobotRunnerError("load the policy before reading runtime facts")
        return self._facts

    def load(self) -> VLABenchSmolVLARunner:
        """Load the config, policy, and all four processor pipelines exactly once."""

        if self.loaded:
            return self
        bindings = self._bindings or load_native_bindings()
        self._bindings = bindings

        config = bindings.pre_trained_config.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
        config.pretrained_path = self.checkpoint
        config.pretrained_revision = self.revision
        config.device = self.device
        if self.backbone_path is not None:
            config.vlm_model_name = self.backbone_path

        environment_config = bindings.vlabench_env_config(task=self.task)
        rename_map = dict(VLABENCH_CAMERA_RENAME_MAP)
        policy = bindings.make_policy(
            cfg=config,
            env_cfg=environment_config,
            rename_map=rename_map,
        )
        policy.eval()

        actual_device = str(getattr(policy.config, "device", self.device))
        preprocessor_overrides = {
            "device_processor": {"device": actual_device},
            "rename_observations_processor": {"rename_map": rename_map},
        }
        if self.backbone_path is not None:
            preprocessor_overrides["tokenizer_processor"] = {
                "tokenizer_name": self.backbone_path,
            }
        processor_kwargs: dict[str, Any] = {
            "policy_cfg": config,
            "pretrained_path": config.pretrained_path,
            "preprocessor_overrides": preprocessor_overrides,
        }
        if self.revision is not None:
            processor_kwargs["pretrained_revision"] = self.revision
        preprocessor, postprocessor = bindings.make_pre_post_processors(**processor_kwargs)
        environment_preprocessor, environment_postprocessor = bindings.make_env_pre_post_processors(
            env_cfg=environment_config,
            policy_cfg=config,
        )

        self._config = config
        self._environment_config = environment_config
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._environment_preprocessor = environment_preprocessor
        self._environment_postprocessor = environment_postprocessor
        self._facts = SmolVLARuntimeFacts(
            checkpoint=self.checkpoint,
            lerobot_version=bindings.version,
            device=actual_device,
            input_feature_shapes=feature_shapes(config.input_features),
            output_feature_shapes=feature_shapes(config.output_features),
            environment_feature_shapes=feature_shapes(environment_config.features),
            preprocessor_stat_shapes=processor_stat_shapes(preprocessor),
            postprocessor_stat_shapes=processor_stat_shapes(postprocessor),
            camera_rename_map=rename_map,
            chunk_size=getattr(config, "chunk_size", None),
            n_action_steps=getattr(config, "n_action_steps", None),
            model_parameter_count=parameter_count(policy),
        )
        return self

    def reset_action_queue(self) -> None:
        """Discard cached action chunks at an episode or subgoal boundary."""

        self.load()
        reset = getattr(self._policy, "reset", None)
        if not callable(reset):
            raise LeRobotRunnerError("loaded policy does not expose reset()")
        reset()
        self._active_prompt = None

    def select_action(
        self,
        observation: Mapping[str, Any],
        prompt: str,
        *,
        reset_action_queue: bool = False,
    ) -> SmolVLAAction:
        """Select one 7-D action from a raw, non-vectorized observation."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        self.load()

        if reset_action_queue or (
            self._active_prompt is not None and prompt != self._active_prompt
        ):
            reset = getattr(self._policy, "reset", None)
            if not callable(reset):
                raise LeRobotRunnerError("loaded policy does not expose reset()")
            reset()
        self._active_prompt = prompt

        assert self._bindings is not None
        assert self._config is not None
        assert self._preprocessor is not None
        assert self._postprocessor is not None
        assert self._environment_preprocessor is not None
        assert self._environment_postprocessor is not None

        batch = self._bindings.preprocess_observation(dict(observation))
        batch["task"] = [prompt]
        batch = self._environment_preprocessor(batch)
        batch = self._preprocessor(batch)

        queue = policy_action_queue(self._policy)
        try:
            generated_chunk = queue is not None and len(queue) == 0
        except TypeError:
            generated_chunk = False

        actual_device = str(getattr(self._config, "device", self.device))
        if self._bindings.synchronize is not None:
            self._bindings.synchronize(actual_device)
        start_s = self._clock()
        with self._bindings.inference_context(
            actual_device,
            bool(getattr(self._config, "use_amp", False)),
        ):
            action = self._policy.select_action(batch)
        if self._bindings.synchronize is not None:
            self._bindings.synchronize(actual_device)
        inference_latency_s = self._clock() - start_s
        if inference_latency_s < 0:
            raise LeRobotRunnerError("clock moved backwards while measuring inference")

        queue = policy_action_queue(self._policy)
        try:
            queue_remaining = None if queue is None else len(queue)
        except TypeError:
            queue_remaining = None
        action = self._postprocessor(action)
        action_transition = self._environment_postprocessor({"action": action})
        if not isinstance(action_transition, Mapping) or "action" not in action_transition:
            raise LeRobotRunnerError("environment postprocessor did not return an action")
        return SmolVLAAction(
            action=one_cpu_action(action_transition["action"]),
            inference_latency_s=inference_latency_s,
            generated_chunk=generated_chunk,
            queue_remaining=queue_remaining,
        )


__all__ = ["VLABenchSmolVLARunner"]
