"""Argument parsing and override application for navigation."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import DEFAULT_CONFIG_PATH
from .settings import BACKENDS, Go2NavigationAppConfig
from .validation import robot_service_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run language-conditioned robot navigation.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--instruction")
    parser.add_argument("--episode-id")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--max-runtime-s", type=float)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument("--lease-duration-s", type=float)
    parser.add_argument(
        "--session-mode",
        choices=("auto", "continuous", "reactive"),
        help="Select image-reactive pulses or continuous waypoint tracking.",
    )
    parser.add_argument("--camera-url")
    parser.add_argument("--control-url")
    parser.add_argument(
        "--robot-host",
        "--robot-ip",
        dest="robot_host",
        help="Robot hostname or IP; derives both HTTP service URLs.",
    )
    parser.add_argument("--camera-port", type=int)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--camera-token")
    parser.add_argument("--control-token")
    parser.add_argument("--control-timeout-s", type=float)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-api-key")
    parser.add_argument("--streamvln-root")
    parser.add_argument("--internvla-root")
    parser.add_argument("--navila-root")
    parser.add_argument("--internvla-variant", choices=("dualvln", "navdp"))
    parser.add_argument("--model-path", "--checkpoint", dest="model_path")
    parser.add_argument("--device")
    parser.add_argument("--cuda-memory-fraction", type=float)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--max-vx", type=float)
    parser.add_argument("--max-vy", type=float)
    parser.add_argument("--max-yaw-rate", type=float)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable robot velocity writes; the default is observation-only.",
    )
    return parser


def _run_overrides(config: Go2NavigationAppConfig, args: argparse.Namespace):
    values = {
        "instruction": args.instruction,
        "episode_id": args.episode_id,
        "max_runtime_s": args.max_runtime_s,
        "control_hz": args.control_hz,
        "lease_duration_s": args.lease_duration_s,
        "session_mode": args.session_mode,
    }
    return replace(config.run, **{key: value for key, value in values.items() if value is not None})


def _policy_overrides(config: Go2NavigationAppConfig, args: argparse.Namespace):
    values = {
        "backend": args.backend,
        "qwen_base_url": args.qwen_base_url,
        "qwen_model": args.qwen_model,
        "qwen_api_key": args.qwen_api_key,
        "streamvln_root": args.streamvln_root,
        "internvla_root": args.internvla_root,
        "internvla_variant": args.internvla_variant,
        "navila_root": args.navila_root,
    }
    policy = replace(
        config.policy,
        **{key: value for key, value in values.items() if value is not None},
    )
    if args.device is not None:
        policy = replace(
            policy,
            streamvln_device=args.device,
            internvla_device=args.device,
            navila_device=args.device,
        )
    if args.model_path is not None:
        field = {
            "qwen": "qwen_model",
            "streamvln": "streamvln_model_path",
            "internvla": "internvla_model_path",
            "navila": "navila_model_path",
        }[policy.backend]
        policy = replace(policy, **{field: args.model_path})
    if args.cuda_memory_fraction is not None:
        field = (
            "navila_cuda_memory_fraction" if policy.backend == "navila" else "cuda_memory_fraction"
        )
        policy = replace(policy, **{field: args.cuda_memory_fraction})
    if args.max_new_tokens is not None:
        field = "navila_max_new_tokens" if policy.backend == "navila" else "max_new_tokens"
        policy = replace(policy, **{field: args.max_new_tokens})
    if args.allow_download:
        field = "navila_local_files_only" if policy.backend == "navila" else "local_files_only"
        policy = replace(policy, **{field: False})
    if args.no_warmup:
        policy = replace(policy, warmup=False)
    return policy


def _go2_overrides(config: Go2NavigationAppConfig, args: argparse.Namespace):
    go2 = config.go2
    if args.robot_host is not None:
        if args.camera_url is not None or args.control_url is not None:
            raise ValueError("--robot-host cannot be combined with explicit service URLs")
        go2 = replace(
            go2,
            camera_url=robot_service_url(args.robot_host, args.camera_port or 8765),
            control_url=robot_service_url(args.robot_host, args.control_port or 8080),
        )
    elif args.camera_port is not None or args.control_port is not None:
        raise ValueError("--camera-port and --control-port require --robot-host")
    values = {
        "camera_url": args.camera_url,
        "control_url": args.control_url,
        "camera_token": args.camera_token,
        "control_token": args.control_token,
        "control_timeout_s": args.control_timeout_s,
        "max_abs_vx_mps": args.max_vx,
        "max_abs_vy_mps": args.max_vy,
        "max_abs_yaw_rate_rps": args.max_yaw_rate,
    }
    return replace(go2, **{key: value for key, value in values.items() if value is not None})


def apply_cli_overrides(
    config: Go2NavigationAppConfig,
    args: argparse.Namespace,
) -> Go2NavigationAppConfig:
    return Go2NavigationAppConfig(
        run=_run_overrides(config, args),
        policy=_policy_overrides(config, args),
        go2=_go2_overrides(config, args),
    )


__all__ = ["apply_cli_overrides", "build_parser"]
