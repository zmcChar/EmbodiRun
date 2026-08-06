"""Compose a navigation provider with the bounded Go2 camera and controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from embodied_runtime.integrations.serving import InferenceProvider
from embodied_runtime.integrations.serving.navigation import (
    InternVLANavigationProvider,
    QwenNavigationProvider,
    StreamVLNNavigationProvider,
)
from embodied_runtime.robots.go2 import (
    Go2CameraClient,
    Go2ControlClient,
    Go2Limits,
    WaypointFollowerConfig,
    WorldWaypointFollower,
)

DEFAULT_CONFIG_PATH = Path("configs/go2_navigation.toml")
BACKENDS = ("qwen", "streamvln", "internvla")
GO2_CONTROL_HARD_LIMITS = Go2Limits(0.35, 0.35, 0.7)
DEFAULT_GO2_CAMERA_URL = "http://127.0.0.1:8765"
DEFAULT_GO2_CONTROL_URL = "http://127.0.0.1:8080"


def _nonempty(value: object, name: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        suffix = " or omitted" if optional else ""
        raise ValueError(f"{name} must be a non-empty string{suffix}")
    return str(value).strip()


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _network_port(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def _robot_service_url(host: str, port: int) -> str:
    host = str(_nonempty(host, "robot_host"))
    if "://" in host or any(character in host for character in "/?#@"):
        raise ValueError("robot_host must be a hostname or IP address, not a URL")
    port = _network_port(port, "robot service port")
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"http://{authority}:{port}"


@dataclass(frozen=True, slots=True)
class RunSettings:
    instruction: str | None = None
    episode_id: str = "go2-navigation"
    max_runtime_s: float = 120.0
    control_hz: float = 10.0
    lease_duration_s: float = 10.0
    max_events: int = 256
    session_mode: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "instruction", _nonempty(self.instruction, "instruction", optional=True)
        )
        object.__setattr__(self, "episode_id", _nonempty(self.episode_id, "episode_id"))
        for name in ("max_runtime_s", "control_hz", "lease_duration_s"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if isinstance(self.max_events, bool) or not isinstance(self.max_events, int):
            raise TypeError("max_events must be a positive integer")
        if self.max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if self.session_mode not in {"auto", "continuous", "reactive"}:
            raise ValueError("session_mode must be 'auto', 'continuous', or 'reactive'")


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    backend: str = "streamvln"
    qwen_base_url: str = "http://127.0.0.1:15003/v1"
    qwen_model: str = "qwen3.5-9b"
    qwen_api_key: str | None = None
    qwen_timeout_s: float = 60.0
    streamvln_root: str | None = None
    streamvln_model_path: str | None = None
    streamvln_device: str = "cuda:0"
    cuda_memory_fraction: float | None = 0.45
    max_new_tokens: int = 64
    local_files_only: bool = True
    warmup: bool = True
    internvla_root: str | None = None
    internvla_model_path: str | None = None
    internvla_variant: str = "dualvln"
    internvla_device: str = "cuda:0"

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        for name in ("qwen_base_url", "qwen_model", "streamvln_device", "internvla_device"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
        for name in (
            "qwen_api_key",
            "streamvln_root",
            "streamvln_model_path",
            "internvla_root",
            "internvla_model_path",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name, optional=True))
        object.__setattr__(self, "qwen_timeout_s", _positive(self.qwen_timeout_s, "qwen_timeout_s"))
        fraction = self.cuda_memory_fraction
        if fraction is not None:
            fraction = _positive(fraction, "cuda_memory_fraction")
            if fraction > 1.0:
                raise ValueError("cuda_memory_fraction must not exceed 1")
            object.__setattr__(self, "cuda_memory_fraction", fraction)
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            raise TypeError("max_new_tokens must be a positive integer")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.internvla_variant not in {"dualvln", "navdp"}:
            raise ValueError("internvla_variant must be 'dualvln' or 'navdp'")
        if not isinstance(self.local_files_only, bool) or not isinstance(self.warmup, bool):
            raise TypeError("local_files_only and warmup must be booleans")


@dataclass(frozen=True, slots=True)
class Go2Settings:
    camera_url: str = DEFAULT_GO2_CAMERA_URL
    control_url: str = DEFAULT_GO2_CONTROL_URL
    camera_token: str | None = None
    control_token: str | None = None
    camera_timeout_s: float = 2.0
    control_timeout_s: float = 1.0
    max_abs_vx_mps: float = 0.35
    max_abs_vy_mps: float = 0.35
    max_abs_yaw_rate_rps: float = 0.7
    position_tolerance_m: float = 0.08
    yaw_tolerance_rad: float = 0.08

    def __post_init__(self) -> None:
        for name in ("camera_url", "control_url"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name))
            parsed = urlsplit(getattr(self, name))
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"{name} must be an absolute HTTP(S) URL")
        for name in ("camera_token", "control_token"):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name, optional=True))
        for name in (
            "camera_timeout_s",
            "control_timeout_s",
            "max_abs_vx_mps",
            "max_abs_vy_mps",
            "max_abs_yaw_rate_rps",
            "position_tolerance_m",
            "yaw_tolerance_rad",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        hard = GO2_CONTROL_HARD_LIMITS
        requested = (self.max_abs_vx_mps, self.max_abs_vy_mps, self.max_abs_yaw_rate_rps)
        ceilings = (hard.max_abs_vx_mps, hard.max_abs_vy_mps, hard.max_abs_yaw_rate_rps)
        if any(value > ceiling for value, ceiling in zip(requested, ceilings)):
            raise ValueError("Go2 limits cannot exceed vx=0.35, vy=0.35, yaw_rate=0.7")


@dataclass(frozen=True, slots=True)
class Go2NavigationAppConfig:
    run: RunSettings = RunSettings()
    provider: ProviderSettings = ProviderSettings()
    go2: Go2Settings = Go2Settings()


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a TOML table")
    return value


def load_config(path: str | Path) -> Go2NavigationAppConfig:
    """Load the checked-in TOML schema without constructing a model or robot client."""

    with Path(path).expanduser().open("rb") as stream:
        raw = tomllib.load(stream)
    session = _table(raw, "session")
    provider = _table(raw, "provider")
    qwen = _table(raw, "qwen")
    streamvln = _table(raw, "streamvln")
    internvla = _table(raw, "internvla")
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
        provider=ProviderSettings(
            backend=str(provider.get("backend", "streamvln")),
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


def build_navigation_provider(settings: ProviderSettings) -> InferenceProvider:
    """Construct only the selected backend; the existing Qwen server is never managed."""

    if settings.backend == "qwen":
        return QwenNavigationProvider(
            base_url=settings.qwen_base_url,
            model=settings.qwen_model,
            api_key=settings.qwen_api_key,
            timeout_s=settings.qwen_timeout_s,
        )
    if settings.backend == "streamvln":
        if settings.streamvln_root is None or settings.streamvln_model_path is None:
            raise ValueError("StreamVLN requires repository and checkpoint paths")
        return StreamVLNNavigationProvider(
            streamvln_root=settings.streamvln_root,
            model_path=settings.streamvln_model_path,
            device=settings.streamvln_device,
            cuda_memory_fraction=settings.cuda_memory_fraction,
            max_new_tokens=settings.max_new_tokens,
            local_files_only=settings.local_files_only,
            warmup=settings.warmup,
        )
    return InternVLANavigationProvider(
        variant=settings.internvla_variant,
        model_path=settings.internvla_model_path,
        device=settings.internvla_device,
        internnav_root=settings.internvla_root,
    )


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(output: Callable[[str], None], payload: Mapping[str, object]) -> None:
    output(json.dumps(_json_value(payload), ensure_ascii=False, sort_keys=True))


async def run_go2_navigation(
    config: Go2NavigationAppConfig,
    *,
    execute: bool = False,
    output: Callable[[str], None] = print,
    provider_factory: Callable[[ProviderSettings], InferenceProvider] = build_navigation_provider,
    camera_factory: Callable[..., Any] = Go2CameraClient,
    control_factory: Callable[..., Any] = Go2ControlClient,
    session_factory: Callable[..., Any] | None = None,
) -> object:
    """Build and run one session. ``execute=False`` is observation-only."""

    if config.run.instruction is None:
        raise ValueError("navigation instruction is required (use --instruction)")
    from embodied_runtime.robots.go2.reactive_session import (
        Go2ReactiveNavigationSession,
        Go2ReactiveSessionConfig,
    )
    from embodied_runtime.robots.go2.session import (  # local import keeps CLI inspection light
        Go2NavigationSession,
        Go2NavigationSessionConfig,
    )

    provider = provider_factory(config.provider)
    try:
        camera = camera_factory(
            config.go2.camera_url,
            token=config.go2.camera_token,
            timeout_s=config.go2.camera_timeout_s,
        )
        control = control_factory(
            config.go2.control_url,
            token=config.go2.control_token,
            timeout_s=config.go2.control_timeout_s,
        )
        limits = Go2Limits(
            config.go2.max_abs_vx_mps,
            config.go2.max_abs_vy_mps,
            config.go2.max_abs_yaw_rate_rps,
        )
        follower = WorldWaypointFollower(
            WaypointFollowerConfig(
                position_tolerance_m=config.go2.position_tolerance_m,
                yaw_tolerance_rad=config.go2.yaw_tolerance_rad,
                limits=limits,
            )
        )

        def event_sink(event: object) -> None:
            payload = _json_value(event)
            if not isinstance(payload, Mapping):
                payload = {"event": payload}
            _emit(output, payload)

        session_mode = config.run.session_mode
        if session_mode == "auto":
            session_mode = "reactive" if config.provider.backend == "streamvln" else "continuous"
        if session_mode == "reactive":
            session_class = session_factory or Go2ReactiveNavigationSession
            session = session_class(
                provider,
                camera,
                control,
                config=Go2ReactiveSessionConfig(
                    max_runtime_s=config.run.max_runtime_s,
                    execute=execute,
                    linear_speed_mps=min(0.30, limits.max_abs_vx_mps, limits.max_abs_vy_mps),
                    yaw_rate_rps=min(0.60, limits.max_abs_yaw_rate_rps),
                    max_events=config.run.max_events,
                    limits=limits,
                ),
                event_sink=event_sink,
            )
        else:
            session_class = session_factory or Go2NavigationSession
            session = session_class(
                provider,
                camera,
                control,
                follower=follower,
                config=Go2NavigationSessionConfig(
                    control_hz=config.run.control_hz,
                    max_runtime_s=config.run.max_runtime_s,
                    execute=execute,
                    lease_duration_s=config.run.lease_duration_s,
                    max_events=config.run.max_events,
                ),
                event_sink=event_sink,
            )
        _emit(
            output,
            {
                "kind": "navigation_start",
                "backend": config.provider.backend,
                "episode_id": config.run.episode_id,
                "mode": "execute" if execute else "dry-run",
                "session_mode": session_mode,
            },
        )
        result = await session.run(config.run.instruction, episode_id=config.run.episode_id)
    finally:
        await provider.aclose()
    summary = _json_value(result)
    if not isinstance(summary, Mapping):
        summary = {"result": summary}
    summary = dict(summary)
    summary.pop("events", None)
    _emit(
        output,
        {
            "kind": "navigation_summary",
            "ok": True,
            "backend": config.provider.backend,
            "mode": "execute" if execute else "dry-run",
            **summary,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--instruction")
    parser.add_argument("--episode-id")
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--max-runtime-s", type=float)
    parser.add_argument("--control-hz", type=float)
    parser.add_argument("--lease-duration-s", type=float)
    parser.add_argument(
        "--session-mode",
        choices=("auto", "continuous", "reactive"),
        help="Use image-reactive pulses or odometry-based continuous tracking; auto selects reactive for StreamVLN.",
    )
    parser.add_argument("--camera-url")
    parser.add_argument("--control-url")
    parser.add_argument(
        "--robot-host",
        "--robot-ip",
        dest="robot_host",
        help="Robot hostname or IP; derives both HTTP service URLs.",
    )
    parser.add_argument("--camera-port", type=int, default=None)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--camera-token")
    parser.add_argument("--control-token")
    parser.add_argument("--control-timeout-s", type=float)
    parser.add_argument("--qwen-base-url")
    parser.add_argument("--qwen-model")
    parser.add_argument("--qwen-api-key")
    parser.add_argument("--streamvln-root")
    parser.add_argument("--internvla-root")
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
        help="Enable Go2 velocity writes. Without this flag the session is dry-run.",
    )
    return parser


def apply_cli_overrides(
    config: Go2NavigationAppConfig, args: argparse.Namespace
) -> Go2NavigationAppConfig:
    run = config.run
    provider = config.provider
    go2 = config.go2
    run_values = {
        "instruction": args.instruction,
        "episode_id": args.episode_id,
        "max_runtime_s": args.max_runtime_s,
        "control_hz": args.control_hz,
        "lease_duration_s": args.lease_duration_s,
        "session_mode": args.session_mode,
    }
    run = replace(run, **{key: value for key, value in run_values.items() if value is not None})
    provider_values = {
        "backend": args.backend,
        "qwen_base_url": args.qwen_base_url,
        "qwen_model": args.qwen_model,
        "qwen_api_key": args.qwen_api_key,
        "streamvln_root": args.streamvln_root,
        "internvla_root": args.internvla_root,
        "internvla_variant": args.internvla_variant,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "max_new_tokens": args.max_new_tokens,
    }
    provider = replace(
        provider,
        **{key: value for key, value in provider_values.items() if value is not None},
    )
    if args.device is not None:
        provider = replace(
            provider,
            streamvln_device=args.device,
            internvla_device=args.device,
        )
    if args.model_path is not None:
        if provider.backend == "qwen":
            provider = replace(provider, qwen_model=args.model_path)
        elif provider.backend == "streamvln":
            provider = replace(provider, streamvln_model_path=args.model_path)
        else:
            provider = replace(provider, internvla_model_path=args.model_path)
    if args.allow_download:
        provider = replace(provider, local_files_only=False)
    if args.no_warmup:
        provider = replace(provider, warmup=False)
    if args.robot_host is not None:
        if args.camera_url is not None or args.control_url is not None:
            raise ValueError("--robot-host cannot be combined with explicit service URLs")
        camera_port = 8765 if args.camera_port is None else args.camera_port
        control_port = 8080 if args.control_port is None else args.control_port
        go2 = replace(
            go2,
            camera_url=_robot_service_url(args.robot_host, camera_port),
            control_url=_robot_service_url(args.robot_host, control_port),
        )
    elif args.camera_port is not None or args.control_port is not None:
        raise ValueError("--camera-port and --control-port require --robot-host")
    go2_values = {
        "camera_url": args.camera_url,
        "control_url": args.control_url,
        "camera_token": args.camera_token,
        "control_token": args.control_token,
        "control_timeout_s": args.control_timeout_s,
        "max_abs_vx_mps": args.max_vx,
        "max_abs_vy_mps": args.max_vy,
        "max_abs_yaw_rate_rps": args.max_yaw_rate,
    }
    go2 = replace(go2, **{key: value for key, value in go2_values.items() if value is not None})
    return Go2NavigationAppConfig(run=run, provider=provider, go2=go2)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = apply_cli_overrides(load_config(args.config), args)
        asyncio.run(run_go2_navigation(config, execute=args.execute))
    except KeyboardInterrupt:
        print(json.dumps({"kind": "navigation_summary", "ok": False, "reason": "interrupted"}))
        return 130
    except Exception as error:  # noqa: BLE001 - command boundary reports structured failure
        print(
            json.dumps(
                {
                    "kind": "navigation_summary",
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    return 0


__all__ = [
    "BACKENDS",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_GO2_CAMERA_URL",
    "DEFAULT_GO2_CONTROL_URL",
    "GO2_CONTROL_HARD_LIMITS",
    "Go2NavigationAppConfig",
    "Go2Settings",
    "ProviderSettings",
    "RunSettings",
    "apply_cli_overrides",
    "build_navigation_provider",
    "build_parser",
    "load_config",
    "main",
    "run_go2_navigation",
]


if __name__ == "__main__":
    raise SystemExit(main())
