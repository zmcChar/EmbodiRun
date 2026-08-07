from __future__ import annotations

import json

import pytest

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.vla.pi05.configuration import manual_pi05_config


def test_manual_config_parser_handles_lerobot_checkpoint_shape(tmp_path) -> None:
    pytest.importorskip("lerobot")
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {
                    "observation.images.base": {
                        "type": "VISUAL",
                        "shape": [3, 224, 224],
                    },
                    "observation.state": {"type": "STATE", "shape": [32]},
                },
                "output_features": {
                    "action": {"type": "ACTION", "shape": [32]},
                },
                "device": "mps",
                "dtype": "float32",
                "image_resolution": [224, 224],
                "optimizer_betas": [0.9, 0.95],
                "use_amp": False,
                "push_to_hub": True,
            }
        ),
        encoding="utf-8",
    )

    config = manual_pi05_config(
        config_file,
        load_device="cpu",
        load_dtype="bfloat16",
    )

    assert config.device == "cpu"
    assert config.dtype == "bfloat16"
    assert config.image_resolution == (224, 224)
    assert config.optimizer_betas == (0.9, 0.95)
    assert config.input_features["observation.images.base"].shape == (3, 224, 224)
    assert config.output_features["action"].shape == (32,)


def test_manual_config_parser_rejects_unknown_semantic_fields(tmp_path) -> None:
    pytest.importorskip("lerobot")
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "type": "pi05",
                "input_features": {},
                "output_features": {},
                "future_field": "cannot-assume-descriptive",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackageError, match="future_field"):
        manual_pi05_config(
            config_file,
            load_device="cpu",
            load_dtype="bfloat16",
        )
