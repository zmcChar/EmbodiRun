from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.models.errors import ModelPackageError
from embodied_runtime.models.request import RawRequest
from embodied_runtime.models.vla.pi05.processing_pi05 import (
    LANGUAGE_ATTENTION_MASK,
    LANGUAGE_TOKENS,
    Pi05Processor,
)


@dataclass
class _Config:
    image_features: dict
    image_resolution: tuple[int, int] = (4, 4)
    max_action_dim: int = 2
    chunk_size: int = 3
    num_inference_steps: int = 4


class _FakePolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = _Config(
            image_features={
                "observation.images.base": object(),
                "observation.images.wrist": object(),
            }
        )

    def _preprocess_images(self, batch):
        images = []
        masks = []
        first = next(value for key, value in batch.items() if key.startswith("observation.images."))
        batch_size = first.shape[0]
        for name in self.config.image_features:
            if name in batch:
                images.append(batch[name].to(self.anchor.device, torch.float32) * 2 - 1)
                masks.append(torch.ones(batch_size, dtype=torch.bool, device=self.anchor.device))
            else:
                images.append(torch.full_like(first, -1.0, dtype=torch.float32))
                masks.append(torch.zeros(batch_size, dtype=torch.bool, device=self.anchor.device))
        return images, masks


def test_lerobot_style_batch_is_prepared() -> None:
    processor = Pi05Processor(_FakePolicy())
    payload = processor.from_lerobot_batch(
        {
            "observation.images.base": torch.rand(2, 3, 4, 4),
            "observation.images.wrist": torch.rand(2, 3, 4, 4),
            LANGUAGE_TOKENS: torch.ones(2, 5, dtype=torch.long),
            LANGUAGE_ATTENTION_MASK: torch.ones(2, 5, dtype=torch.bool),
        }
    )
    assert len(payload["images"]) == 2
    assert payload["images"][0].shape == (2, 3, 4, 4)
    assert payload["tokens"].shape == (2, 5)
    assert payload["token_masks"].dtype == torch.bool


def test_generic_requests_support_ordered_camera_tensor() -> None:
    processor = Pi05Processor(_FakePolicy())
    requests = [
        RawRequest(
            observation={
                "images": torch.rand(2, 3, 4, 4),
                "instruction_tokens": torch.arange(5),
            }
        ),
        RawRequest(
            observation={
                "images": torch.rand(2, 3, 4, 4),
                "instruction_tokens": torch.arange(5),
            }
        ),
    ]
    payload = processor.from_requests(requests)
    assert payload["tokens"].shape == (2, 5)
    assert all(image.shape[0] == 2 for image in payload["images"])


def test_prompt_only_request_is_rejected_with_boundary_hint() -> None:
    processor = Pi05Processor(_FakePolicy())
    with pytest.raises(ModelPackageError, match="pre-tokenized"):
        processor.from_requests(
            [
                RawRequest(
                    observation={"images": torch.rand(2, 3, 4, 4)},
                    prompt="pick up the block",
                )
            ]
        )


def test_synthetic_batch_uses_checkpoint_shapes() -> None:
    processor = Pi05Processor(_FakePolicy())
    payload = processor.synthetic_batch(batch_size=1, language_length=7, seed=4)
    assert len(payload["images"]) == 2
    assert payload["images"][0].shape == (1, 3, 4, 4)
    assert payload["tokens"].shape == (1, 7)
