from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import embodied_runtime.integrations.navigation.vvla.checkpoint as checkpoint_module
from embodied_runtime.integrations.navigation.vvla import (
    verify_activevln_checkpoint,
    verify_activevln_source,
)
from embodied_runtime.policies.navigation.errors import NavigationPolicyError

SCRIPT = Path(__file__).parents[3] / "examples/vvla_activevln_habitat.py"
SPEC = importlib.util.spec_from_file_location("vvla_activevln_habitat", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
habitat_example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(habitat_example)


def test_source_verification_invokes_vvla_pin_verifier(monkeypatch, tmp_path: Path) -> None:
    vvla_root = tmp_path / "vvla"
    verifier = vvla_root / "scripts/activevln/verify_source_pin.py"
    verifier.parent.mkdir(parents=True)
    verifier.touch()
    source_root = tmp_path / "ActiveVLN"
    source_root.mkdir()
    calls: list[object] = []

    def fake_run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="ActiveVLN source lock is valid\n")

    monkeypatch.setattr(checkpoint_module.subprocess, "run", fake_run)

    verify_activevln_source(vvla_root, source_root)

    assert calls == [
        (
            [
                sys.executable,
                str(verifier.resolve()),
                "--source-root",
                str(source_root.resolve()),
            ],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


def test_source_verification_fails_closed_on_pin_mismatch(monkeypatch, tmp_path: Path) -> None:
    verifier = tmp_path / "vvla/scripts/activevln/verify_source_pin.py"
    verifier.parent.mkdir(parents=True)
    verifier.touch()
    source_root = tmp_path / "ActiveVLN"
    source_root.mkdir()
    monkeypatch.setattr(
        checkpoint_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, stdout="", stderr="ActiveVLN HEAD mismatch"
        ),
    )

    with pytest.raises(NavigationPolicyError, match="source verification failed.*HEAD mismatch"):
        verify_activevln_source(tmp_path / "vvla", source_root)


def test_checkpoint_verification_invokes_vvla_pin_verifier(monkeypatch, tmp_path: Path) -> None:
    vvla_root = tmp_path / "vvla"
    verifier = vvla_root / "scripts/activevln/verify_source_pin.py"
    verifier.parent.mkdir(parents=True)
    verifier.touch()
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    calls: list[object] = []

    def fake_run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 0, stdout="ActiveVLN source lock is valid\n")

    monkeypatch.setattr(checkpoint_module.subprocess, "run", fake_run)

    verify_activevln_checkpoint(vvla_root, checkpoint_root)

    assert calls == [
        (
            [
                sys.executable,
                str(verifier.resolve()),
                "--checkpoint-root",
                str(checkpoint_root.resolve()),
            ],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


def test_checkpoint_verification_distinguishes_remote_ids_from_local_paths(
    tmp_path: Path,
) -> None:
    verify_activevln_checkpoint(tmp_path / "missing-vvla", "Arvil/remote-checkpoint")

    with pytest.raises(NavigationPolicyError, match="directory does not exist"):
        verify_activevln_checkpoint(tmp_path / "missing-vvla", tmp_path / "missing-checkpoint")


def test_habitat_main_verifies_source_before_asset_or_runtime_load(monkeypatch) -> None:
    args = SimpleNamespace(
        max_episodes=1,
        max_turns=1,
        checkpoint=None,
        preflight_only=True,
        runtime="vvla",
        attention="eager",
        vvla_root=Path("/vvla"),
        activevln_root=Path("/ActiveVLN"),
    )
    events: list[object] = []
    monkeypatch.setattr(
        habitat_example,
        "_parser",
        lambda: SimpleNamespace(parse_args=lambda: args),
    )
    monkeypatch.setattr(
        habitat_example,
        "verify_activevln_source",
        lambda vvla_root, source_root: events.append(("verify", vvla_root, source_root)),
    )

    def stop_at_assets(received) -> None:
        events.append(("assets", received))
        raise SystemExit("stop after source verification")

    monkeypatch.setattr(habitat_example, "_require_assets", stop_at_assets)

    with pytest.raises(SystemExit, match="stop after source verification"):
        habitat_example.main()

    assert events == [
        ("verify", args.vvla_root, args.activevln_root),
        ("assets", args),
    ]


def _runtime_args(runtime: str) -> SimpleNamespace:
    return SimpleNamespace(
        runtime=runtime,
        vvla_root=Path("/vvla"),
        checkpoint=Path("/models/activevln"),
        device="cuda:0",
        dtype="bfloat16",
        attention="eager",
        max_new_tokens=24,
        max_context=4096,
    )


def test_habitat_cli_defaults_to_vvla_and_accepts_transformers() -> None:
    required = [
        "--activevln-root",
        "/ActiveVLN",
        "--dataset-root",
        "/data/r2r",
        "--scenes-dir",
        "/data/mp3d",
    ]

    assert habitat_example._parser().parse_args(required).runtime == "vvla"
    assert (
        habitat_example._parser().parse_args([*required, "--runtime", "transformers"]).runtime
        == "transformers"
    )


@pytest.mark.parametrize(
    ("runtime_name", "attribute"),
    [
        ("transformers", "TransformersActiveVLNRuntime"),
        ("vvla", "VvlaActiveVLNRuntime"),
    ],
)
def test_habitat_runtime_selector_preserves_backend_specific_options(
    monkeypatch, runtime_name: str, attribute: str
) -> None:
    captured = {}
    sentinel = object()

    def runtime_type(**options):
        captured.update(options)
        return sentinel

    monkeypatch.setattr(habitat_example, attribute, runtime_type)

    assert habitat_example._runtime_from_args(_runtime_args(runtime_name)) is sentinel
    assert captured["checkpoint"] == Path("/models/activevln")
    assert captured["dtype"] == "bfloat16"
    assert captured["attention"] == "eager"
    assert captured["max_new_tokens"] == 24
    assert captured["max_context"] == 4096
    assert captured.get("env_id") == ("habitat-r2r" if runtime_name == "vvla" else None)


def test_transformers_habitat_driver_uses_strict_actions_and_resets_session() -> None:
    actions = (SimpleNamespace(name="move forward", value=25),)
    prediction = SimpleNamespace(
        text="move forward 25cm",
        actions=actions,
        latency_ms=12.5,
        token_ids=(10, 11),
    )
    calls = []

    class Runtime:
        def predict(self, rgb, instruction, *, episode_id):
            calls.append((rgb, instruction, episode_id))
            return prediction

        def reset(self):
            calls.append("reset")

    parsed = SimpleNamespace(valid=True, raw_text=prediction.text, actions=actions)
    driver = habitat_example._TransformersActiveVLNHabitatDriver(Runtime(), lambda text: parsed)
    observation = SimpleNamespace(
        images=np.full((1, 3, 2, 4), 0.5, dtype=np.float32),
        instruction="Walk to the door.",
    )
    session = SimpleNamespace(env_id=3, episode_id="episode-7", rollout_id=0)

    turn = driver.act(observation, session)

    rgb, instruction, episode_id = calls[0]
    assert rgb.shape == (2, 4, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb == 128)
    assert instruction == "Walk to the door."
    assert episode_id == "3:episode-7:0"
    assert turn.action is parsed
    assert turn.raw_tokens == ()
    assert turn.timing == {"e2e_ms": 12.5}
    assert turn.metadata["token_ids"] == (10, 11)

    driver.reset_session(session)
    assert calls[-1] == "reset"


def test_transformers_habitat_driver_rejects_unsafe_text_before_upstream_parser() -> None:
    actions = (SimpleNamespace(name="move forward", value=25),)

    class Runtime:
        @staticmethod
        def predict(rgb, instruction, *, episode_id):
            del rgb, instruction, episode_id
            return SimpleNamespace(
                text="please do not move forward 25cm",
                actions=actions,
                latency_ms=1.0,
                token_ids=(1,),
            )

    parser_calls = []
    driver = habitat_example._TransformersActiveVLNHabitatDriver(
        Runtime(), lambda text: parser_calls.append(text)
    )
    observation = SimpleNamespace(
        images=np.zeros((1, 3, 2, 2), dtype=np.float32),
        instruction="Walk to the door.",
    )
    session = SimpleNamespace(env_id=0, episode_id="episode", rollout_id=0)

    with pytest.raises(NavigationPolicyError, match="non-canonical"):
        driver.act(observation, session)

    assert parser_calls == []
