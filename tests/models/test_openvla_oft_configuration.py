from __future__ import annotations

import json

import pytest

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.vla.openvla_oft.checkpoint import resolve_openvla_oft_checkpoint
from embodied_runtime.models.vla.openvla_oft.configuration import load_model_config


def _write_checkpoint(tmp_path, *, num_images: int = 1) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {"vocab_size": 32064},
                "use_fused_vision_backbone": True,
                "num_images_in_input": num_images,
                "timm_model_ids": ["dinov2", "siglip"],
                "image_sizes": [224, 224],
                "timm_override_act_layers": [None, None],
                "unnorm_key": "bridge",
                "num_action_chunks": 8,
                "pad_to_multiple_of": 64,
                "n_action_bins": 256,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "bridge_no_noops": {
                    "action": {
                        "q01": [-1.0, -2.0],
                        "q99": [1.0, 2.0],
                        "mask": [True, False],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()


def test_checkpoint_and_model_config_are_pure_and_exact(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    root = resolve_openvla_oft_checkpoint(tmp_path)
    config = load_model_config(root, action_dim=2, action_horizon=None)

    assert root == tmp_path.resolve()
    assert config.statistics_key == "bridge_no_noops"
    assert config.action_dim == 2
    assert config.action_horizon == 8
    assert config.action_vocab_size == 32000
    assert config.padded_vocab_size == 32064
    assert config.n_action_bins == 256
    assert config.action_mask == [True, False]
    assert config.timm_model_ids == ("dinov2", "siglip")


def test_model_config_rejects_multi_image_before_backbone_construction(tmp_path) -> None:
    _write_checkpoint(tmp_path, num_images=2)

    with pytest.raises(ModelPackageError, match="exactly one input image"):
        load_model_config(tmp_path, action_dim=None, action_horizon=None)


def test_model_config_rejects_action_dimension_mismatch(tmp_path) -> None:
    _write_checkpoint(tmp_path)

    with pytest.raises(ModelPackageError, match="requested action_dim=7"):
        load_model_config(tmp_path, action_dim=7, action_horizon=None)
