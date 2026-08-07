from __future__ import annotations

import importlib.metadata

import pytest

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.vla.smolvla.checkpoint import (
    checkpoint_source,
    normalization_tensor_keys,
    validate_vlm_source,
)
from embodied_runtime.models.vla.smolvla.constants import (
    DEFAULT_SMOLVLA_REVISION,
    EXPECTED_LEROBOT_VERSION,
)
from embodied_runtime.models.vla.smolvla.dependencies import require_lerobot_033


def test_pinned_revision_and_normalization_keys_are_exact() -> None:
    assert DEFAULT_SMOLVLA_REVISION == "3326b100334ffc0a0bd1ec27e3afb1cfa2a6000c"
    assert normalization_tensor_keys("so100") == {
        "observation.state": {
            "mean": "normalize_inputs.so100_buffer_observation_state.mean",
            "std": "normalize_inputs.so100_buffer_observation_state.std",
        },
        "action": {
            "mean": "unnormalize_outputs.so100_buffer_action.mean",
            "std": "unnormalize_outputs.so100_buffer_action.std",
        },
    }


def test_normalization_keys_reject_empty_variant() -> None:
    with pytest.raises(ValueError, match="stats_variant must not be empty"):
        normalization_tensor_keys("  ")


def test_local_checkpoint_and_vlm_sources_are_validated(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").touch()
    weights = checkpoint / "model.safetensors"
    weights.touch()

    source, local_weights = checkpoint_source(checkpoint)

    assert source == str(checkpoint)
    assert local_weights == weights
    assert validate_vlm_source(checkpoint, load_weights=True) == str(checkpoint.resolve())


def test_lerobot_version_mismatch_is_rejected_before_policy_import(monkeypatch) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "0.5.1")

    with pytest.raises(
        ModelPackageError,
        match=f"requires lerobot=={EXPECTED_LEROBOT_VERSION}, found 0.5.1",
    ):
        require_lerobot_033()
