from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/prepare_navila_eager_overlay.py"
SPEC = importlib.util.spec_from_file_location("prepare_navila_eager_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
overlay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(overlay)


def _pinned_source() -> str:
    return (
        f"{overlay._MODEL_OUTPUTS_IMPORT}\n"
        f"{overlay._LEGACY_ROTARY_BLOCK}\n"
        f"{overlay._FLASH_MASK_BLOCK}\n"
        f"{overlay._FLASH_ATTENTION_BLOCK}"
    )


def test_eager_patch_replaces_only_the_pinned_attention_constructs() -> None:
    patched = overlay.eager_modeling_llama(_pinned_source())

    assert overlay._EAGER_ATTENTION_BLOCK in patched
    assert overlay._EAGER_ROTARY_BLOCK in patched
    assert overlay._EAGER_MASK_IMPORT in patched
    assert overlay._EAGER_MASK_BLOCK in patched
    assert overlay._FLASH_ATTENTION_BLOCK not in patched


def test_eager_patch_rejects_an_unknown_or_changed_upstream_source() -> None:
    with pytest.raises(RuntimeError, match="SOURCE_MISMATCH"):
        overlay.eager_modeling_llama(_pinned_source().replace("LlamaFlashAttention2", "Changed"))


def test_overlay_installs_patched_transformers_and_fail_closed_flash_stub(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "site-packages/transformers"
    source_llama = source / "models/llama/modeling_llama.py"
    source_llama.parent.mkdir(parents=True)
    source_llama.write_text(_pinned_source(), encoding="utf-8")
    extra = source / "models/llama/extra.py"
    extra.write_text("VALUE = 1\n", encoding="utf-8")
    target.mkdir(parents=True)

    overlay.install_eager_overlay(source, target)

    assert (target / "models/llama/modeling_llama.py").read_text(
        encoding="utf-8"
    ) == overlay.eager_modeling_llama(_pinned_source())
    assert (target / "models/llama/extra.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    flash_stub = target.parent / "flash_attn/__init__.py"
    assert "FLASH_ATTENTION_DISABLED_FOR_NAVILA_EAGER" in flash_stub.read_text(encoding="utf-8")
    interface_stub = target.parent / "flash_attn/flash_attn_interface.py"
    interface_source = interface_stub.read_text(encoding="utf-8")
    assert "flash_attn_unpadded_qkvpacked_func = _disabled" in interface_source
    assert "flash_attn_varlen_qkvpacked_func = _disabled" in interface_source
