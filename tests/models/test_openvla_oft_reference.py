from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.models.vla.openvla_oft import OpenVLAOFTReferenceModule
from embodied_runtime.models.vla.openvla_oft.head import CategoricalActionHead
from embodied_runtime.models.vla.openvla_oft.loading import (
    default_dtype,
    materialize_language_buffers,
)


class _TinyVision(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.anchor = torch.nn.Parameter(torch.ones((), dtype=torch.bfloat16))
        self.last_pixel_dtype = None

    def forward(self, pixel_values):
        self.last_pixel_dtype = pixel_values.dtype
        batch_size = pixel_values.shape[0]
        patches = torch.arange(
            2 * self.hidden_size,
            device=pixel_values.device,
            dtype=self.anchor.dtype,
        ).reshape(1, 2, self.hidden_size)
        return patches.expand(batch_size, -1, -1)


class _IdentityProjector(torch.nn.Module):
    def forward(self, patches):
        return patches


class _TinyLanguageCore(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)


class _PositionEncodedLanguageModel(torch.nn.Module):
    """Emit distinct categorical maxima at the action-prediction positions."""

    def __init__(self, vocab_size: int, hidden_size: int, n_tokens: int) -> None:
        super().__init__()
        self.model = _TinyLanguageCore(vocab_size, hidden_size)
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.vocab_size = vocab_size
        self.n_tokens = n_tokens
        self.last_inputs_embeds = None
        self.last_attention_mask = None
        self.last_position_ids = None

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        **options,
    ):
        del input_ids, options
        self.last_inputs_embeds = inputs_embeds
        self.last_attention_mask = attention_mask
        self.last_position_ids = position_ids
        batch_size, sequence_length, _ = inputs_embeds.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -100.0,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        # The native OFT causal read is [-n_tokens-1:-1]. Give that region
        # [8, 9, 10, 11], and give the final (excluded) position a different
        # maximum so an off-by-one slice is observable.
        prediction_tokens = (8, 9, 10, 11, 8)
        start = sequence_length - self.n_tokens - 1
        for offset, token_id in enumerate(prediction_tokens):
            logits[:, start + offset, token_id] = 100.0
        return SimpleNamespace(logits=logits)


def test_reference_module_reads_space_then_previous_action_positions() -> None:
    action_dim = 2
    action_horizon = 2
    n_tokens = action_dim * action_horizon
    vocab_size = 12
    hidden_size = 4
    head = CategoricalActionHead(
        vocab_size=vocab_size,
        n_action_bins=4,
        action_dim=action_dim,
        num_action_chunks=action_horizon,
        q01=[-1.0, -1.0],
        q99=[1.0, 1.0],
        mask=[False, False],
    )
    language_model = _PositionEncodedLanguageModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        n_tokens=n_tokens,
    )
    vision = _TinyVision(hidden_size)
    module = OpenVLAOFTReferenceModule(
        vision,
        _IdentityProjector(),
        language_model,
        head,
    )

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]], dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "pixel_values": torch.ones(2, 6, 2, 2),
    }
    outputs = module(inputs)

    expected_tokens = torch.tensor([[8, 9, 10, 11], [8, 9, 10, 11]])
    assert torch.equal(outputs["action_tokens"], expected_tokens)
    assert outputs["actions"].shape == (2, action_horizon, action_dim)
    torch.testing.assert_close(
        outputs["actions"],
        head.tokens_to_actions(expected_tokens),
    )

    # Three text tokens + two visual patches + four zero-content action
    # queries. The final query still participates in the Llama forward but its
    # logits do not predict another action.
    assert language_model.last_inputs_embeds.shape == (2, 9, hidden_size)
    assert language_model.last_attention_mask.shape == (2, 9)
    assert language_model.last_position_ids.shape == (2, 9)
    assert torch.count_nonzero(language_model.last_inputs_embeds[:, -n_tokens:]) == 0
    assert vision.last_pixel_dtype is torch.bfloat16


def test_meta_initialized_llama_rotary_buffers_are_materialized() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    with torch.device("meta"), default_dtype(torch.bfloat16):
        language_model = transformers.LlamaForCausalLM(config)

    assert any(value.is_meta for _, value in language_model.named_buffers())

    materialize_language_buffers(
        language_model,
        config,
        load_device="cpu",
    )

    assert not any(value.is_meta for _, value in language_model.named_buffers())
