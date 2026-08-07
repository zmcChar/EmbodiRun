"""Input processing for the pinned LeRobot SmolVLA deployment adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ...errors import ModelPackageError
from ...request import RawRequest
from .modeling_smolvla import LANGUAGE_ATTENTION_MASK, LANGUAGE_TOKENS

OBSERVATION_STATE = "observation.state"


def _as_tensor(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def _retain_batch_one(
    value: Any,
    *,
    expected_ndim: int,
    description: str,
) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim == expected_ndim:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != expected_ndim + 1 or tensor.shape[0] != 1:
        raise ModelPackageError(
            f"one SmolVLA {description} must retain B=1; got {tuple(tensor.shape)}"
        )
    return tensor


class SmolVLAProcessor:
    """Convert raw robot observations into one fixed-shape model payload."""

    def __init__(self, lerobot_policy: torch.nn.Module) -> None:
        self.policy = lerobot_policy

    @property
    def image_features(self) -> tuple[str, ...]:
        return tuple(self.policy.config.image_features)

    @property
    def tokenizer(self) -> Any:
        tokenizer = getattr(self.policy, "language_tokenizer", None)
        if tokenizer is None:
            model = getattr(self.policy, "model", None)
            vlm = getattr(model, "vlm_with_expert", None)
            processor = getattr(vlm, "processor", None)
            tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ModelPackageError("loaded SmolVLA policy does not expose a tokenizer")
        return tokenizer

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        task = prompt if prompt.endswith("\n") else f"{prompt}\n"
        encoded = self.tokenizer(
            [task],
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=int(self.policy.config.tokenizer_max_length),
            return_tensors="pt",
        )
        return (
            _retain_batch_one(
                encoded["input_ids"],
                expected_ndim=1,
                description="language token sequence",
            ).to(torch.long),
            _retain_batch_one(
                encoded["attention_mask"],
                expected_ndim=1,
                description="language attention mask",
            ).to(torch.bool),
        )

    def preprocess_one(self, request: RawRequest) -> dict[str, torch.Tensor]:
        observation = request.observation
        payload: dict[str, torch.Tensor] = {}

        ordered_images = observation.get("images")
        image_tensor = None if ordered_images is None else _as_tensor(ordered_images)
        if image_tensor is not None and image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        for index, feature_name in enumerate(self.image_features):
            if feature_name in observation:
                image = observation[feature_name]
            elif image_tensor is not None and index < image_tensor.shape[0]:
                image = image_tensor[index]
            else:
                raise ModelPackageError(
                    f"SmolVLA request is missing image feature {feature_name!r}"
                )
            payload[feature_name] = _retain_batch_one(
                image,
                expected_ndim=3,
                description=f"image {feature_name!r}",
            ).to(torch.float32)

        state = observation.get(OBSERVATION_STATE, observation.get("state"))
        if state is None:
            raise ModelPackageError(f"SmolVLA request is missing {OBSERVATION_STATE!r} or 'state'")
        payload[OBSERVATION_STATE] = _retain_batch_one(
            state,
            expected_ndim=1,
            description="state",
        ).to(torch.float32)

        tokens = observation.get(
            LANGUAGE_TOKENS,
            observation.get("instruction_tokens", observation.get("tokens")),
        )
        if tokens is None:
            prompt = request.prompt or observation.get("task")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ModelPackageError("SmolVLA request needs a prompt or pre-tokenized language")
            token_tensor, mask_tensor = self._tokenize(prompt)
        else:
            token_tensor = _retain_batch_one(
                tokens,
                expected_ndim=1,
                description="language token sequence",
            ).to(torch.long)
            mask = observation.get(
                LANGUAGE_ATTENTION_MASK,
                observation.get("token_masks"),
            )
            mask_tensor = (
                torch.ones_like(token_tensor, dtype=torch.bool)
                if mask is None
                else _retain_batch_one(
                    mask,
                    expected_ndim=1,
                    description="language attention mask",
                ).to(torch.bool)
            )
            if mask_tensor.shape != token_tensor.shape:
                raise ModelPackageError(
                    "SmolVLA language tokens and attention mask must have equal shape"
                )
        payload[LANGUAGE_TOKENS] = token_tensor
        payload[LANGUAGE_ATTENTION_MASK] = mask_tensor
        return payload

    def from_lerobot_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Validate an already-batched LeRobot observation mapping."""

        payload: dict[str, torch.Tensor] = {}
        for feature_name in self.image_features:
            if feature_name not in batch:
                raise ModelPackageError(f"SmolVLA batch is missing image feature {feature_name!r}")
            image = _as_tensor(batch[feature_name])
            if image.ndim != 4:
                raise ModelPackageError(f"SmolVLA image {feature_name!r} must be [B,C,H,W]")
            payload[feature_name] = image.to(torch.float32)

        for key, dtype in (
            (OBSERVATION_STATE, torch.float32),
            (LANGUAGE_TOKENS, torch.long),
            (LANGUAGE_ATTENTION_MASK, torch.bool),
        ):
            if key not in batch:
                raise ModelPackageError(f"SmolVLA batch is missing {key!r}")
            tensor = _as_tensor(batch[key])
            if tensor.ndim != 2:
                raise ModelPackageError(f"SmolVLA batch field {key!r} must be two-dimensional")
            payload[key] = tensor.to(dtype)

        batch_sizes = {int(value.shape[0]) for value in payload.values()}
        if len(batch_sizes) != 1:
            raise ModelPackageError("SmolVLA batch fields have inconsistent batch sizes")
        if payload[LANGUAGE_TOKENS].shape != payload[LANGUAGE_ATTENTION_MASK].shape:
            raise ModelPackageError(
                "SmolVLA language tokens and attention mask must have equal shape"
            )
        return payload

    def synthetic_batch(
        self,
        *,
        batch_size: int = 1,
        language_length: int = 48,
        seed: int = 0,
    ) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if not 0 < language_length <= int(self.policy.config.tokenizer_max_length):
            raise ValueError("language_length must be between one and the tokenizer maximum")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        payload: dict[str, torch.Tensor] = {}
        for feature_name, feature in self.policy.config.image_features.items():
            shape = tuple(int(dimension) for dimension in feature.shape)
            payload[feature_name] = torch.rand(
                (batch_size, *shape),
                generator=generator,
            )
        state_feature = self.policy.config.robot_state_feature
        if state_feature is None:
            raise ModelPackageError("SmolVLA config has no robot state feature")
        state_dim = int(state_feature.shape[0])
        payload[OBSERVATION_STATE] = (
            torch.linspace(
                -0.25,
                0.25,
                state_dim,
            )
            .expand(batch_size, -1)
            .clone()
        )
        payload[LANGUAGE_TOKENS] = torch.randint(
            0,
            1024,
            (batch_size, language_length),
            generator=generator,
        )
        payload[LANGUAGE_ATTENTION_MASK] = torch.ones(
            batch_size,
            language_length,
            dtype=torch.bool,
        )
        return payload


__all__ = ["OBSERVATION_STATE", "SmolVLAProcessor"]
