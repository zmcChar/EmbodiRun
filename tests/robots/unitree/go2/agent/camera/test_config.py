from __future__ import annotations

import argparse

import pytest

from embodied_runtime.robots.unitree.go2.agent.camera.argument_types import camera_profile
from embodied_runtime.robots.unitree.go2.agent.camera.arguments import parse_args
from embodied_runtime.robots.unitree.go2.agent.camera.settings import validate_bind_token

TOKEN = "camera-token-" + "x" * 32


def test_profile_cli_and_environment_are_resolved_without_machine_addresses() -> None:
    from_environment = parse_args(
        [],
        environ={
            "GO2_CAMERA_PROFILE": "848x480@30",
            "GO2_CAMERA_HOST": "127.0.0.1",
            "GO2_CAMERA_PORT": "9000",
            "GO2_CAMERA_SERIAL": "ABC123",
            "GO2_CAMERA_JPEG_FPS": "8",
        },
    )
    assert from_environment.profile == "848x480@30"
    assert from_environment.port == 9000
    assert from_environment.serial == "ABC123"
    assert from_environment.jpeg_fps == 8.0

    overridden = parse_args(
        ["--profile", "640x360@15", "--width", "1280", "--fps", "10"],
        environ={},
    )
    assert (overridden.width, overridden.height, overridden.fps) == (1280, 360, 10)
    assert overridden.depth_width == 1280
    assert overridden.depth_height == 360


def test_realsense_requires_explicit_calibration_and_scale() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--backend", "realsense"], environ={})
    config = parse_args(
        [
            "--backend",
            "realsense",
            "--depth-calibrated",
            "--depth-scale",
            "0.001",
        ],
        environ={},
    )
    assert config.backend == "realsense"
    assert config.depth_scale == 0.001
    assert config.depth_calibrated


def test_external_bind_requires_strong_ascii_bearer_token() -> None:
    with pytest.raises(ValueError, match="required"):
        validate_bind_token("0.0.0.0", None)
    with pytest.raises(ValueError, match="at least 32"):
        validate_bind_token("192.0.2.10", "short")
    with pytest.raises(ValueError, match="ASCII"):
        validate_bind_token("192.0.2.10", "密" * 32)
    with pytest.raises(ValueError, match="whitespace"):
        validate_bind_token("192.0.2.10", "x" * 31 + " ")
    validate_bind_token("192.0.2.10", TOKEN)
    validate_bind_token("127.0.0.1", None)


def test_token_and_depth_scale_can_come_from_environment() -> None:
    config = parse_args(
        ["--backend", "realsense"],
        environ={
            "GO2_CAMERA_HOST": "0.0.0.0",
            "GO2_CAMERA_TOKEN": TOKEN,
            "GO2_CAMERA_DEPTH_CALIBRATED": "true",
            "GO2_CAMERA_DEPTH_SCALE": "0.001",
            "GO2_CAMERA_PROFILE": "640x360@15",
        },
    )
    assert config.host == "0.0.0.0"
    assert config.token == TOKEN
    assert config.depth_scale == 0.001


def test_python38_compatible_boolean_flags_support_positive_and_negative_forms() -> None:
    enabled = parse_args(
        ["--access-log", "--rgb-depth-aligned", "--depth-calibrated", "--depth-scale", "0.001"],
        environ={},
    )
    assert enabled.access_log is True
    assert enabled.rgb_depth_alignment_claimed is True

    disabled = parse_args(
        ["--no-access-log", "--no-depth-calibrated"],
        environ={
            "GO2_CAMERA_ACCESS_LOG": "true",
            "GO2_CAMERA_DEPTH_CALIBRATED": "true",
        },
    )
    assert disabled.access_log is False
    assert disabled.depth_calibrated is False


@pytest.mark.parametrize("invalid", ["640x360", "640*360@15", "0x360@15"])
def test_profile_parser_rejects_ambiguous_values(invalid: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        camera_profile(invalid)
