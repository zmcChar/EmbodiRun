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
    DEFAULT_ACTIVEVLN_CHECKPOINT,
    DEFAULT_ACTIVEVLN_REVISION,
    DEFAULT_GO2_CAMERA_URL,
    DEFAULT_GO2_CONTROL_URL,
    DEFAULT_VLLM_OMNI_URL,
    DEFAULT_VVLA_ROOT,
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


def _policy_model(policy: Mapping[str, Any]) -> str:
    """Resolve the canonical model key and its legacy backend alias."""

    model = policy.get("model")
    backend = policy.get("backend")
    if model is not None and backend is not None and str(model) != str(backend):
        raise ValueError("policy.model and legacy policy.backend must match when both are set")
    return str(model if model is not None else backend if backend is not None else "streamvln")


def load_config(path: str | Path) -> Go2NavigationAppConfig:
    """Load settings without constructing a model or robot client."""

    with Path(path).expanduser().open("rb") as stream:
        raw = tomllib.load(stream)
    session = _table(raw, "session")
    policy = _table(raw, "policy")
    vllm_omni = _table(raw, "vllm_omni")
    qwen = _table(raw, "qwen")
    streamvln = _table(raw, "streamvln")
    internvla = _table(raw, "internvla")
    navila = _table(raw, "navila")
    activevln = _table(raw, "activevln")
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
            backend=_policy_model(policy),
            runtime=str(policy.get("runtime", "transformers")),
            vllm_omni_url=str(vllm_omni.get("url", DEFAULT_VLLM_OMNI_URL)),
            vllm_omni_timeout_s=float(vllm_omni.get("timeout_s", 120.0)),
            vllm_omni_session_id=vllm_omni.get("session_id"),
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
            vvla_root=str(activevln.get("repository", DEFAULT_VVLA_ROOT)),
            activevln_checkpoint=str(activevln.get("checkpoint", DEFAULT_ACTIVEVLN_CHECKPOINT)),
            activevln_revision=str(activevln.get("revision", DEFAULT_ACTIVEVLN_REVISION)),
            activevln_device=str(activevln.get("device", "cuda:0")),
            activevln_dtype=str(activevln.get("dtype", "bfloat16")),
            activevln_attention=str(activevln.get("attention", "eager")),
            activevln_max_new_tokens=int(activevln.get("max_new_tokens", 64)),
            activevln_max_context=int(activevln.get("max_context", 32768)),
            activevln_allow_download=bool(activevln.get("allow_download", False)),
            activevln_do_sample=bool(activevln.get("do_sample", False)),
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
