"""TOML loading for the navigation application."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from .settings import (
    DEFAULT_GO2_CAMERA_URL,
    DEFAULT_GO2_CONTROL_URL,
    Go2NavigationAppConfig,
    Go2Settings,
    PolicySettings,
    RunSettings,
)

DEFAULT_CONFIG_PATH = Path("configs/go2_navigation.toml")


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table")
    return value


def load_config(path: str | Path) -> Go2NavigationAppConfig:
    """Load settings without constructing a model or robot client."""

    with Path(path).expanduser().open("rb") as stream:
        raw = tomllib.load(stream)
    session = _table(raw, "session")
    policy = _table(raw, "policy")
    qwen = _table(raw, "qwen")
    streamvln = _table(raw, "streamvln")
    internvla = _table(raw, "internvla")
    navila = _table(raw, "navila")
    go2 = _table(raw, "go2")
    follower = _table(go2, "follower")

    return Go2NavigationAppConfig(
        run=RunSettings(
            instruction=session.get("instruction"),
            episode_id=str(session.get("episode_id", "go2-navigation")),
            max_runtime_s=float(session.get("max_runtime_s", 120.0)),
            control_hz=float(session.get("control_hz", go2.get("control_hz", 10.0))),
            lease_duration_s=float(session.get("lease_duration_s", 10.0)),
            max_events=int(session.get("max_events", 256)),
            session_mode=str(session.get("mode", "auto")),
        ),
        policy=PolicySettings(
            backend=str(policy.get("backend", "streamvln")),
            qwen_base_url=str(qwen.get("base_url", "http://127.0.0.1:15003/v1")),
            qwen_model=str(qwen.get("model", "qwen3.5-9b")),
            qwen_api_key=qwen.get("api_key") or os.environ.get("QWEN_API_KEY"),
            qwen_timeout_s=float(qwen.get("timeout_s", 60.0)),
            streamvln_root=streamvln.get("repository"),
            streamvln_model_path=streamvln.get("checkpoint"),
            streamvln_device=str(streamvln.get("device", "cuda:0")),
            cuda_memory_fraction=streamvln.get("cuda_memory_fraction", 0.45),
            max_new_tokens=int(streamvln.get("max_new_tokens", 64)),
            local_files_only=bool(streamvln.get("local_files_only", True)),
            warmup=bool(streamvln.get("warmup", True)),
            internvla_root=internvla.get("repository"),
            internvla_model_path=internvla.get("checkpoint"),
            internvla_variant=str(internvla.get("variant", "dualvln")),
            internvla_device=str(internvla.get("device", "cuda:0")),
            navila_root=navila.get("repository"),
            navila_model_path=navila.get("checkpoint"),
            navila_device=str(navila.get("device", "cuda:0")),
            navila_cuda_memory_fraction=navila.get("cuda_memory_fraction"),
            navila_max_new_tokens=int(navila.get("max_new_tokens", 32)),
            navila_local_files_only=bool(navila.get("local_files_only", True)),
        ),
        go2=Go2Settings(
            camera_url=str(go2.get("camera_url", DEFAULT_GO2_CAMERA_URL)),
            control_url=str(go2.get("control_url", DEFAULT_GO2_CONTROL_URL)),
            camera_token=go2.get("camera_token") or os.environ.get("GO2_CAMERA_TOKEN"),
            control_token=go2.get("control_token") or os.environ.get("GO2_API_TOKEN"),
            camera_timeout_s=float(go2.get("camera_timeout_s", 2.0)),
            control_timeout_s=float(go2.get("control_timeout_s", 1.0)),
            max_abs_vx_mps=float(follower.get("max_abs_vx_mps", 0.35)),
            max_abs_vy_mps=float(follower.get("max_abs_vy_mps", 0.35)),
            max_abs_yaw_rate_rps=float(follower.get("max_abs_yaw_rate_rps", 0.7)),
            position_tolerance_m=float(follower.get("position_tolerance_m", 0.08)),
            yaw_tolerance_rad=float(follower.get("yaw_tolerance_rad", 0.08)),
        ),
    )


__all__ = ["DEFAULT_CONFIG_PATH", "load_config"]
