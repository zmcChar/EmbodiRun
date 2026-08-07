from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_runtime.models.vla.navila import (
    NAVILA_STOP_STRING,
    NaVILAAction,
    NaVILAConfig,
    NaVILANativeOutputError,
    NaVILAPrediction,
    NaVILAPrimitive,
    NaVILARuntime,
    build_navila_prompt,
    parse_navila_action,
    sample_episode_frames,
)

ROOT = Path(__file__).parents[2]


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "NaVILA"
    for relative in ("llava/constants.py", "llava/mm_utils.py", "llava/model/builder.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    return root


@pytest.mark.parametrize(
    ("text", "primitive", "magnitude", "unit"),
    [
        ("The action is STOP.", NaVILAPrimitive.STOP, None, None),
        ("The action is move forward 50 cm.", NaVILAPrimitive.MOVE_FORWARD, 50, "cm"),
        ("The action is turn left 30 degrees.", NaVILAPrimitive.TURN_LEFT, 30, "degree"),
        ("The action is turn right 45 degree.", NaVILAPrimitive.TURN_RIGHT, 45, "degree"),
        ("move forward", NaVILAPrimitive.MOVE_FORWARD, 25, "cm"),
        ("turn left", NaVILAPrimitive.TURN_LEFT, 15, "degree"),
    ],
)
def test_native_action_parser_golden_outputs(text, primitive, magnitude, unit) -> None:
    action = parse_navila_action(text)
    assert action == NaVILAAction(primitive, magnitude, unit)


@pytest.mark.parametrize(
    "text",
    [
        "continue ahead",
        "move forward 25 cm and stop",
        "turn left 15 degrees then turn right 15 degrees",
        "move forward 38 cm",
        "turn left 20 degrees",
        "move forward 0 cm",
        "turn right 495 degrees",
    ],
)
def test_native_action_parser_fails_closed(text: str) -> None:
    with pytest.raises(NaVILANativeOutputError):
        parse_navila_action(text)


def test_prompt_matches_released_eight_image_template() -> None:
    prompt = build_navila_prompt("Walk to the red chair")
    assert prompt.count("<image>") == 8
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>system")
    assert prompt.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    assert NAVILA_STOP_STRING == "<|eot_id|>"
    assert 'Your assigned task is: "Walk to the red chair"' in prompt


def test_full_episode_sampling_keeps_uniform_history_and_latest() -> None:
    assert sample_episode_frames(tuple(range(1, 5))) == (1, 2, 3, 4)
    assert sample_episode_frames(tuple(range(10))) == (0, 1, 2, 3, 5, 6, 7, 9)


def test_config_is_local_and_pins_real_smoke_compatible_options(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    config = NaVILAConfig(root, "a8cheng/navila-llama3-8b-8f", device="cuda:1")
    assert config.navila_root == root.resolve()
    assert config.device == "cuda:1"
    assert config.dtype == "float16"
    assert config.attention_backend == "eager"
    assert config.max_new_tokens == 32


class _Evaluator:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, frames, instruction):
        self.calls.append((frames, instruction))
        return NaVILAPrediction(
            NaVILAAction(NaVILAPrimitive.MOVE_FORWARD, 25, "cm"),
            "move forward 25 cm",
            (1,),
            0.1,
        )


def test_runtime_loads_once_and_serializes_prediction(tmp_path: Path) -> None:
    root = _source_tree(tmp_path)
    evaluator = _Evaluator()
    loads = 0

    def materializer(config):
        nonlocal loads
        loads += 1
        return SimpleNamespace(model="model", evaluator=evaluator)

    runtime = NaVILARuntime(navila_root=root, materializer=materializer)
    assert not runtime.loaded
    assert runtime.load() is runtime
    result = runtime.predict(("frame",), "go")
    assert result.action.primitive is NaVILAPrimitive.MOVE_FORWARD
    assert loads == 1
    assert evaluator.calls == [(("frame",), "go")]
    runtime.close()
    with pytest.raises(Exception, match="closed"):
        runtime.predict(("frame",), "go")


def test_navila_import_is_dependency_lazy() -> None:
    code = """
import sys
before = set(sys.modules)
import embodied_runtime.models.vla.navila
loaded = {name.split('.', 1)[0] for name in set(sys.modules) - before}
assert not loaded & {'torch', 'transformers', 'llava', 'PIL', 'numpy'}, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
