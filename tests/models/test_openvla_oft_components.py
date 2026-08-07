from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.request import RawRequest
from embodied_runtime.models.vla.openvla_oft.head import CategoricalActionHead
from embodied_runtime.models.vla.openvla_oft.processing_openvla_oft import (
    PROMPT_TEMPLATE,
    SPACE_TOKEN,
    OpenVLAOFTProcessor,
)


def _head() -> CategoricalActionHead:
    return CategoricalActionHead(
        vocab_size=12,
        n_action_bins=4,
        action_dim=2,
        num_action_chunks=2,
        q01=[0.0, -1.0],
        q99=[2.0, 1.0],
        mask=[True, False],
    )


def _logits() -> torch.Tensor:
    logits = torch.full((1, 4, 14), -5.0)
    logits[..., 0] = 100.0  # invalid text token
    logits[..., 13] = 90.0  # invalid padded-vocabulary token
    logits[..., 8] = 1.0
    logits[..., 9] = 2.0
    logits[..., 10] = 3.0
    logits[..., 11] = 0.0
    return logits


def test_action_head_greedy_masks_non_action_logits() -> None:
    head = _head()

    token_ids, logprob = head.greedy(_logits())

    assert torch.equal(token_ids, torch.full((1, 4), 10))
    assert logprob.shape == token_ids.shape
    assert torch.isfinite(logprob).all()


def test_action_head_sampling_and_logprob_recompute_match() -> None:
    head = _head()
    logits = _logits()

    sampled, behavior_logprob = head.sample(
        logits,
        do_sample=True,
        temperature=0.7,
        top_k=1,
        generator=torch.Generator().manual_seed(7),
    )
    recomputed = head.logprob(
        logits,
        sampled,
        temperature=0.7,
        top_k=1,
    )

    assert torch.equal(sampled, torch.full((1, 4), 10))
    assert torch.equal(behavior_logprob, recomputed)
    assert torch.equal(
        recomputed,
        head.recompute_logprob(logits, sampled, temperature=0.7, top_k=1),
    )


def test_action_head_logprob_is_differentiable() -> None:
    head = _head()
    logits = _logits().requires_grad_(True)
    token_ids, _ = head.greedy(logits.detach())

    head.logprob(logits, token_ids).sum().backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_action_head_tokens_to_actions_uses_dataset_mask() -> None:
    head = _head()
    token_ids = torch.tensor([[10, 10, 9, 11]])

    actions = head.tokens_to_actions(token_ids)

    assert actions.shape == (1, 2, 2)
    # Token 10 maps to normalized 0. The first dimension is unnormalized
    # from [-1, 1] to [q01=0, q99=2], while the second remains normalized.
    assert torch.equal(actions[0, 0], torch.tensor([1.0, 0.0]))
    assert actions[0, 1, 0] > actions[0, 0, 0]
    assert actions[0, 1, 1] < actions[0, 0, 1]


class _FakeTokenizer:
    pad_token_id = 7
    bos_token_id = 1

    def __init__(self, token_ids: list[int] | None = None) -> None:
        self.token_ids = token_ids or [1, 2]
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text: str, *, add_special_tokens: bool):
        self.calls.append((text, add_special_tokens))
        return SimpleNamespace(input_ids=list(self.token_ids))


def _processor(
    tokenizer: _FakeTokenizer | None = None,
    *,
    image_size: int = 8,
    max_length: int = 6,
) -> OpenVLAOFTProcessor:
    return OpenVLAOFTProcessor(
        tokenizer or _FakeTokenizer(),
        means=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        stds=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        image_size=image_size,
        max_length=max_length,
    )


def test_processor_returns_batched_inputs_and_prefers_raw_prompt() -> None:
    tokenizer = _FakeTokenizer()
    processor = _processor(tokenizer)
    request = RawRequest(
        observation={
            "image": torch.full((3, 5, 7), 0.25),
            "images": torch.ones(2, 3, 5, 7),
            "instruction": "ignored instruction",
        },
        prompt="Pick UP The Bowl",
    )

    payload = processor.preprocess_one(request)

    assert set(payload) == {"input_ids", "attention_mask", "pixel_values"}
    assert payload["input_ids"].shape == (1, 6)
    assert payload["attention_mask"].shape == (1, 6)
    assert payload["pixel_values"].shape == (1, 6, 8, 8)
    assert payload["input_ids"].dtype == torch.long
    assert payload["attention_mask"].dtype == torch.long
    assert payload["pixel_values"].dtype == torch.float32
    assert payload["input_ids"][0, -1].item() == SPACE_TOKEN
    assert payload["input_ids"][0, 0].item() == tokenizer.bos_token_id
    assert payload["attention_mask"][0, 0].item() == 1
    assert torch.allclose(
        payload["pixel_values"],
        torch.full_like(payload["pixel_values"], 0.25),
    )
    assert tokenizer.calls == [(PROMPT_TEMPLATE.format(task="pick up the bowl"), True)]


def test_processor_uses_first_camera_and_observation_instruction() -> None:
    tokenizer = _FakeTokenizer()
    processor = _processor(tokenizer)
    request = RawRequest(
        observation={
            "images": torch.stack(
                (
                    torch.full((3, 4, 4), 0.2),
                    torch.full((3, 4, 4), 0.8),
                )
            ),
            "instruction": "Move Left",
        }
    )

    payload = processor.preprocess_one(request)

    assert torch.allclose(
        payload["pixel_values"],
        torch.full_like(payload["pixel_values"], 0.2),
    )
    assert tokenizer.calls[0][0] == PROMPT_TEMPLATE.format(task="move left")


def test_processor_preserves_space_token_when_truncating() -> None:
    tokenizer = _FakeTokenizer(list(range(20)))
    processor = _processor(tokenizer, max_length=5)

    payload = processor.preprocess_one(
        RawRequest(
            observation={
                "image": torch.zeros(3, 4, 4),
                "instruction": "long task",
            }
        )
    )

    assert payload["input_ids"].shape == (1, 5)
    assert payload["input_ids"][0, -1].item() == SPACE_TOKEN
    assert torch.equal(payload["attention_mask"], torch.ones(1, 5, dtype=torch.long))


def test_processor_accepts_uint8_image_and_rejects_missing_image() -> None:
    processor = _processor()
    payload = processor.preprocess_one(
        RawRequest(
            observation={
                "image": torch.full((3, 4, 4), 255, dtype=torch.uint8),
                "instruction": "stop",
            }
        )
    )
    assert torch.allclose(
        payload["pixel_values"],
        torch.ones_like(payload["pixel_values"]),
    )

    with pytest.raises(ModelPackageError, match="observation\\['image'\\]"):
        processor.preprocess_one(RawRequest(observation={"instruction": "stop"}))


def test_processor_checkpoint_errors_are_model_specific(tmp_path) -> None:
    with pytest.raises(ModelPackageError, match="preprocessor_config.json"):
        OpenVLAOFTProcessor.from_checkpoint(tmp_path)
