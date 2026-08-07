"""pi0.5 input preparation separated from execution and hardware concerns.

The LeRobot image preprocessing path is reused to preserve exact resize,
padding, empty-camera, and normalization semantics.  Language tokens are
expected to have already been produced by the upstream pi0.5 tokenizer/
processor, matching the LeRobot-style batch boundary.

Portions were refactored from vvla (MIT, Copyright 2026 Longxmas) and interface
with LeRobot/OpenPI code distributed under Apache-2.0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ...errors import ModelPackageError
from ...request import RawRequest

LANGUAGE_TOKENS = "observation.language.tokens"
LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def _as_unbatched_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor


class Pi05Processor:
    """Prepare generic requests or an already-collated LeRobot-style batch."""

    def __init__(self, lerobot_policy: torch.nn.Module) -> None:
        self.policy = lerobot_policy

    @property
    def image_features(self) -> tuple[str, ...]:
        return tuple(self.policy.config.image_features)

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def from_lerobot_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Convert a batched LeRobot mapping into pi0.5 entrypoint inputs."""

        mutable = dict(batch)
        if LANGUAGE_TOKENS not in mutable:
            raise ModelPackageError(
                f"pi0.5 batch is missing {LANGUAGE_TOKENS!r}; prompt-only "
                "tokenization is intentionally outside the ModelAdapter"
            )
        tokens = _as_unbatched_tensor(mutable[LANGUAGE_TOKENS])
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        masks_value = mutable.get(LANGUAGE_ATTENTION_MASK)
        if masks_value is None:
            token_masks = torch.ones_like(tokens, dtype=torch.bool)
        else:
            token_masks = _as_unbatched_tensor(masks_value)
            if token_masks.ndim == 1:
                token_masks = token_masks.unsqueeze(0)
            token_masks = token_masks.to(torch.bool)
        if tokens.shape != token_masks.shape:
            raise ModelPackageError(
                f"pi0.5 token shape {tuple(tokens.shape)} does not match "
                f"attention-mask shape {tuple(token_masks.shape)}"
            )

        images, image_masks = self.policy._preprocess_images(mutable)
        return {
            "images": tuple(images),
            "image_masks": tuple(image_masks),
            "tokens": tokens.to(self.device),
            "token_masks": token_masks.to(self.device),
        }

    def from_requests(self, requests: Sequence[RawRequest]) -> dict[str, Any]:
        """Collate unbatched requests, then apply LeRobot image preprocessing."""

        if not requests:
            raise ModelPackageError("pi0.5 preprocessing needs at least one request")
        batch: dict[str, Any] = {}

        # Accept either named LeRobot image features or one ordered
        # ``images[num_cameras, 3, H, W]`` tensor per request.
        for camera_index, feature_name in enumerate(self.image_features):
            camera_values = []
            present_for_all = True
            for request in requests:
                observation = request.observation
                if feature_name in observation:
                    value = observation[feature_name]
                elif "images" in observation:
                    images = _as_unbatched_tensor(observation["images"])
                    if images.ndim == 3:
                        images = images.unsqueeze(0)
                    if camera_index >= images.shape[0]:
                        present_for_all = False
                        break
                    value = images[camera_index]
                else:
                    present_for_all = False
                    break
                camera_values.append(_as_unbatched_tensor(value))
            if present_for_all:
                batch[feature_name] = torch.stack(camera_values)

        tokens = []
        masks = []
        for request in requests:
            observation = request.observation
            value = observation.get(
                LANGUAGE_TOKENS,
                observation.get("instruction_tokens", observation.get("tokens")),
            )
            if value is None:
                raise ModelPackageError(
                    "pi0.5 requests require pre-tokenized language in "
                    f"{LANGUAGE_TOKENS!r}, 'instruction_tokens', or 'tokens'"
                )
            token = _as_unbatched_tensor(value).to(torch.long)
            if token.ndim != 1:
                raise ModelPackageError(
                    f"each pi0.5 request needs 1-D tokens; got {tuple(token.shape)}"
                )
            mask_value = observation.get(
                LANGUAGE_ATTENTION_MASK,
                observation.get("token_masks"),
            )
            mask = (
                torch.ones_like(token, dtype=torch.bool)
                if mask_value is None
                else _as_unbatched_tensor(mask_value).to(torch.bool)
            )
            if mask.shape != token.shape:
                raise ModelPackageError(
                    f"pi0.5 token shape {tuple(token.shape)} does not match "
                    f"mask shape {tuple(mask.shape)}"
                )
            tokens.append(token)
            masks.append(mask)
        try:
            batch[LANGUAGE_TOKENS] = torch.stack(tokens)
            batch[LANGUAGE_ATTENTION_MASK] = torch.stack(masks)
        except RuntimeError as exc:
            raise ModelPackageError(
                "pi0.5 requests must be padded to one common token length"
            ) from exc
        return self.from_lerobot_batch(batch)

    def synthetic_batch(
        self,
        *,
        batch_size: int = 1,
        language_length: int = 48,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Build a tokenizer-free, checkpoint-shaped smoke-test batch.

        Pixel and token values are synthetic, so the output has no task meaning;
        the method exists to validate real-weight loading and all four entrypoints
        on a GPU before connecting a tokenizer or robot.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if language_length <= 0:
            raise ValueError("language_length must be greater than zero")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        height, width = self.policy.config.image_resolution
        batch: dict[str, Any] = {
            feature: torch.rand(
                batch_size,
                3,
                height,
                width,
                generator=generator,
            )
            for feature in self.image_features
        }
        batch[LANGUAGE_TOKENS] = torch.randint(
            0,
            257_152,
            (batch_size, language_length),
            generator=generator,
        )
        batch[LANGUAGE_ATTENTION_MASK] = torch.ones(
            batch_size,
            language_length,
            dtype=torch.bool,
        )
        return self.from_lerobot_batch(batch)


__all__ = [
    "LANGUAGE_ATTENTION_MASK",
    "LANGUAGE_TOKENS",
    "Pi05Processor",
]
