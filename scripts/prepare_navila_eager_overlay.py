#!/usr/bin/env python3
"""Install NaVILA's pinned Transformers replacement with eager attention."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_FLASH_ATTENTION_BLOCK = """        self.self_attn = (
            # LlamaAttention(config=config)
            LlamaFlashAttention2(config=config)
        )"""
_EAGER_ATTENTION_BLOCK = """        self.self_attn = LlamaAttention(config=config)"""
_LEGACY_ROTARY_BLOCK = """        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)"""
_EAGER_ROTARY_BLOCK = """        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)"""
_MODEL_OUTPUTS_IMPORT = (
    "from ...modeling_outputs import BaseModelOutputWithPast, "
    "CausalLMOutputWithPast, SequenceClassifierOutputWithPast"
)
_EAGER_MASK_IMPORT = (
    "from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask\n"
    + _MODEL_OUTPUTS_IMPORT
)
_FLASH_MASK_BLOCK = """        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None"""
_EAGER_MASK_BLOCK = """        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )"""
_FLASH_STUB = '''"""Import-compatible guard for NaVILA's disabled flash-attention path."""

__version__ = "0+navila-eager-disabled"


def _disabled(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("FLASH_ATTENTION_DISABLED_FOR_NAVILA_EAGER")


def flash_attn_func(*args, window_size=None, **kwargs):
    del window_size
    return _disabled(*args, **kwargs)


def flash_attn_varlen_func(*args, window_size=None, **kwargs):
    del window_size
    return _disabled(*args, **kwargs)
'''
_PADDING_STUB = '''"""Fail-closed symbols imported by NaVILA's Transformers overlay."""


def _disabled(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("FLASH_ATTENTION_DISABLED_FOR_NAVILA_EAGER")


index_first_axis = _disabled
pad_input = _disabled
unpad_input = _disabled
'''
_INTERFACE_STUB = '''"""Fail-closed symbols imported by unused NaVILA vision encoders."""


def _disabled(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("FLASH_ATTENTION_DISABLED_FOR_NAVILA_EAGER")


flash_attn_unpadded_qkvpacked_func = _disabled
flash_attn_varlen_qkvpacked_func = _disabled
_flash_attn_backward = _disabled
_flash_attn_forward = _disabled
_flash_attn_varlen_backward = _disabled
_flash_attn_varlen_forward = _disabled
'''


def eager_modeling_llama(source: str) -> str:
    """Patch only the exact constructs present in the pinned official source."""

    expected = {
        _FLASH_ATTENTION_BLOCK: 1,
        _LEGACY_ROTARY_BLOCK: 1,
        _MODEL_OUTPUTS_IMPORT: 1,
        _FLASH_MASK_BLOCK: 1,
    }
    mismatches = [
        fragment for fragment, count in expected.items() if source.count(fragment) != count
    ]
    if mismatches:
        raise RuntimeError("NAVILA_EAGER_PATCH_SOURCE_MISMATCH")
    return (
        source.replace(_MODEL_OUTPUTS_IMPORT, _EAGER_MASK_IMPORT)
        .replace(_FLASH_ATTENTION_BLOCK, _EAGER_ATTENTION_BLOCK)
        .replace(_LEGACY_ROTARY_BLOCK, _EAGER_ROTARY_BLOCK)
        .replace(_FLASH_MASK_BLOCK, _EAGER_MASK_BLOCK)
    )


def install_eager_overlay(source: Path, transformers_target: Path) -> None:
    source = source.resolve(strict=True)
    transformers_target = transformers_target.resolve(strict=True)
    for source_file in sorted(source.rglob("*.py")):
        relative = source_file.relative_to(source)
        target = transformers_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)

    relative_llama = Path("models/llama/modeling_llama.py")
    llama_source = source / relative_llama
    llama_target = transformers_target / relative_llama
    llama_target.write_text(
        eager_modeling_llama(llama_source.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    flash_stub = transformers_target.parent / "flash_attn"
    flash_stub.mkdir(parents=True, exist_ok=True)
    (flash_stub / "__init__.py").write_text(_FLASH_STUB, encoding="utf-8")
    (flash_stub / "bert_padding.py").write_text(_PADDING_STUB, encoding="utf-8")
    (flash_stub / "flash_attn_interface.py").write_text(
        _INTERFACE_STUB,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--transformers-target", required=True, type=Path)
    args = parser.parse_args()
    install_eager_overlay(args.source, args.transformers_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
