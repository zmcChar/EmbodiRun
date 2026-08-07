"""Lazy loading of the native LeRobot/PyTorch API surface."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from importlib import metadata
from typing import Any

from .values import LeRobotBindings, LeRobotRunnerError


@contextmanager
def _native_inference_context(torch: Any, device: str, use_amp: bool):
    with ExitStack() as stack:
        stack.enter_context(torch.inference_mode())
        if use_amp:
            device_type = str(device).split(":", 1)[0]
            stack.enter_context(torch.autocast(device_type=device_type))
        yield


def _synchronize_native(torch: Any, device: str) -> None:
    if str(device).split(":", 1)[0] == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_native_bindings() -> LeRobotBindings:
    try:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import VLABenchEnv
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies import make_policy, make_pre_post_processors
    except ImportError as error:
        raise LeRobotRunnerError(
            "native VLABench SmolVLA inference requires LeRobot 0.6.x with "
            "its SmolVLA and VLABench dependencies"
        ) from error

    try:
        version = metadata.version("lerobot")
    except metadata.PackageNotFoundError:
        version = None

    return LeRobotBindings(
        pre_trained_config=PreTrainedConfig,
        vlabench_env_config=VLABenchEnv,
        make_policy=make_policy,
        make_pre_post_processors=make_pre_post_processors,
        make_env_pre_post_processors=make_env_pre_post_processors,
        preprocess_observation=preprocess_observation,
        inference_context=lambda device, use_amp: _native_inference_context(torch, device, use_amp),
        version=version,
        synchronize=lambda device: _synchronize_native(torch, device),
    )
