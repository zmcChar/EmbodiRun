from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors = pytest.importorskip("safetensors.torch")

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.vla.openvla_oft.checkpoint import load_checkpoint_components
from embodied_runtime.models.vla.openvla_oft.loading import load_openvla_oft


def test_sharded_loader_assigns_meta_parameters_in_requested_dtype(tmp_path) -> None:
    with torch.device("meta"):
        vision = torch.nn.Linear(2, 2)
        projector = torch.nn.Linear(2, 2)
        language = torch.nn.Linear(2, 2)

    source = {
        "vision_backbone.weight": torch.full((2, 2), 1.0),
        "vision_backbone.bias": torch.full((2,), 2.0),
        "projector.weight": torch.full((2, 2), 3.0),
        "projector.bias": torch.full((2,), 4.0),
        "language_model.weight": torch.full((2, 2), 5.0),
        "language_model.bias": torch.full((2,), 6.0),
    }
    shard_name = "model-00001-of-00001.safetensors"
    safetensors.save_file(source, tmp_path / shard_name)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard_name for name in source}}),
        encoding="utf-8",
    )

    load_checkpoint_components(
        tmp_path,
        vision_backbone=vision,
        projector=projector,
        language_model=language,
        load_device="cpu",
        load_dtype=torch.bfloat16,
        strict=True,
    )

    for module in (vision, projector, language):
        assert not any(parameter.is_meta for parameter in module.parameters())
        assert {parameter.dtype for parameter in module.parameters()} == {torch.bfloat16}
    torch.testing.assert_close(
        language.weight,
        torch.full((2, 2), 5.0, dtype=torch.bfloat16),
    )


def test_first_slice_rejects_multi_image_checkpoint_before_model_build(tmp_path) -> None:
    pytest.importorskip("transformers")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {"vocab_size": 32064},
                "num_images_in_input": 2,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset_statistics.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelPackageError, match="exactly one input image"):
        load_openvla_oft(tmp_path, local_files_only=True)
