"""OpenVLA-OFT image and language preprocessing.

Adapted from ``vvla/policies/openvla_oft/processor_openvla_oft.py`` at vvla
commit ``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT,
Copyright 2026 Longxmas). The BOS relocation follows RLinf's
``PrismaticProcessor`` (Apache-2.0, Copyright 2025 The RLinf Authors).

One :class:`~embodied_runtime.contracts.RawRequest` becomes a tensor-tree whose
leaves retain ``B=1`` so the generic model adapter can collate requests without
knowing OpenVLA-OFT semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from embodied_runtime.contracts import ModelPackageError, RawRequest

PROMPT_TEMPLATE = "In: What action should the robot take to {task}?\nOut: "
SPACE_TOKEN = 29_871


def _require_torchvision():
    try:
        import torchvision.transforms.functional as vision_functional
        from torchvision.transforms import InterpolationMode
    except ImportError as exc:  # pragma: no cover - install-time guard
        raise ModelPackageError(
            "OpenVLA-OFT image preprocessing requires torchvision; "
            "install the OpenVLA-OFT model dependencies"
        ) from exc
    return vision_functional, InterpolationMode


def _token_ids(encoded: Any) -> list[int]:
    value = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ModelPackageError(
                "OpenVLA-OFT tokenizer returned more than one sequence for one prompt"
            )
        value = value[0]
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ModelPackageError("OpenVLA-OFT tokenizer returned invalid input_ids") from exc


class OpenVLAOFTProcessor:
    """Create fixed-shape Llama tokens and fused DINOv2/SigLIP pixels."""

    def __init__(
        self,
        tokenizer: Any,
        means: Any,
        stds: Any,
        *,
        image_size: int = 224,
        max_length: int = 50,
    ) -> None:
        if image_size <= 0 or max_length < 2:
            raise ValueError("image_size must be positive and max_length must be at least two")
        self.tokenizer = tokenizer
        self.means = tuple(torch.as_tensor(value, dtype=torch.float32).flatten() for value in means)
        self.stds = tuple(torch.as_tensor(value, dtype=torch.float32).flatten() for value in stds)
        if len(self.means) != 2 or len(self.stds) != 2:
            raise ValueError(
                "OpenVLA-OFT requires two normalization parameter sets (DINOv2 and SigLIP)"
            )
        if any(tuple(value.shape) != (3,) for value in (*self.means, *self.stds)):
            raise ValueError("each OpenVLA-OFT image mean/std must contain three channels")
        if any(torch.any(value == 0) for value in self.stds):
            raise ValueError("OpenVLA-OFT image standard deviations must be non-zero")
        self.image_size = int(image_size)
        self.max_length = int(max_length)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        max_length: int = 50,
        cache_dir: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> OpenVLAOFTProcessor:
        """Load tokenizer and image normalization metadata from a checkpoint."""

        try:
            from transformers import AutoTokenizer
            from transformers.utils import cached_file
        except ImportError as exc:  # pragma: no cover - install-time guard
            raise ModelPackageError(
                "OpenVLA-OFT tokenization requires transformers; "
                "install the OpenVLA-OFT model dependencies"
            ) from exc

        source = str(Path(checkpoint).expanduser()) if isinstance(checkpoint, Path) else checkpoint
        local = Path(source).expanduser()
        if local.is_dir():
            preprocessor_file = local / "preprocessor_config.json"
            if not preprocessor_file.is_file():
                raise ModelPackageError(
                    "local OpenVLA-OFT checkpoint has no preprocessor_config.json"
                )
        elif local.is_absolute():
            raise ModelPackageError(
                f"local OpenVLA-OFT checkpoint directory does not exist: {local}"
            )
        else:
            try:
                resolved = cached_file(
                    source,
                    "preprocessor_config.json",
                    cache_dir=cache_dir,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                raise ModelPackageError(
                    f"could not resolve OpenVLA-OFT preprocessor_config.json for {source!r}: {exc}"
                ) from exc
            if resolved is None:
                raise ModelPackageError(
                    f"could not resolve OpenVLA-OFT preprocessor_config.json for {source!r}"
                )
            preprocessor_file = Path(resolved)

        try:
            with preprocessor_file.open(encoding="utf-8") as handle:
                config = json.load(handle)
            means = config["means"]
            stds = config["stds"]
            image_size = int(config["input_sizes"][0][-1])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelPackageError(
                f"invalid OpenVLA-OFT preprocessor config {preprocessor_file}: {exc}"
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                padding_side="left",
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise ModelPackageError(
                f"could not load OpenVLA-OFT tokenizer from {source!r}: {exc}"
            ) from exc
        return cls(
            tokenizer,
            means,
            stds,
            image_size=image_size,
            max_length=max_length,
        )

    def _transform_image(self, image: torch.Tensor) -> torch.Tensor:
        """Transform ``[3,H,W]`` RGB in ``[0,1]`` into fused six-channel pixels."""

        vision_functional, interpolation_mode = _require_torchvision()
        resized = vision_functional.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=interpolation_mode.BICUBIC,
            antialias=True,
        )
        normalized = [
            vision_functional.normalize(
                resized,
                mean=mean.tolist(),
                std=std.tolist(),
            )
            for mean, std in zip(self.means, self.stds, strict=True)
        ]
        return torch.cat(normalized, dim=0)

    def _tokenize(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        text = PROMPT_TEMPLATE.format(task=task.lower())
        try:
            encoded = self.tokenizer(text, add_special_tokens=True)
        except Exception as exc:
            raise ModelPackageError(
                f"OpenVLA-OFT tokenizer could not encode the prompt: {exc}"
            ) from exc
        ids = _token_ids(encoded)

        # The action model expects the final real prefix token to be the Llama
        # space token. Reserve that position when truncating long prompts.
        if not ids or ids[-1] != SPACE_TOKEN:
            ids = ids[: self.max_length - 1] + [SPACE_TOKEN]
        elif len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [SPACE_TOKEN]
        padding = self.max_length - len(ids)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        input_ids = [int(pad_token_id)] * padding + ids
        attention_mask = [0] * padding + [1] * len(ids)

        # RLinf's PrismaticProcessor keeps left padding for batch alignment but
        # relocates BOS to position zero so vision patches can always be
        # inserted immediately after it. Remove the original BOS position from
        # the active text and reproduce that checkpoint-facing layout.
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is None and ids:
            bos_token_id = ids[0]
        if padding > 0 and bos_token_id is not None and ids and ids[0] == int(bos_token_id):
            original_bos = padding
            input_ids[original_bos] = int(pad_token_id)
            attention_mask[original_bos] = 0
            input_ids[0] = int(bos_token_id)
            attention_mask[0] = 1
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    @staticmethod
    def _request_image(request: RawRequest) -> torch.Tensor:
        observation = request.observation
        if "image" in observation:
            image = torch.as_tensor(observation["image"])
            if image.ndim == 4 and image.shape[0] == 1:
                image = image[0]
        elif "images" in observation:
            images = torch.as_tensor(observation["images"])
            if images.ndim == 3:
                image = images
            elif images.ndim == 4 and images.shape[0] > 0:
                image = images[0]
            else:
                raise ModelPackageError(
                    "OpenVLA-OFT observation['images'] must have shape [cameras, 3, height, width]"
                )
        else:
            raise ModelPackageError(
                "OpenVLA-OFT requires observation['image'] or observation['images']"
            )
        if image.ndim != 3 or image.shape[0] != 3:
            raise ModelPackageError(
                f"OpenVLA-OFT image must have shape [3, height, width]; got {tuple(image.shape)}"
            )
        if image.dtype == torch.bool:
            raise ModelPackageError("OpenVLA-OFT image must contain numeric RGB values")
        if not torch.is_floating_point(image):
            image = image.to(torch.float32) / 255.0
        else:
            image = image.to(torch.float32)
        if not torch.isfinite(image).all():
            raise ModelPackageError("OpenVLA-OFT image contains NaN or infinity")
        if torch.any(image < 0) or torch.any(image > 1):
            raise ModelPackageError("OpenVLA-OFT floating-point images must use the [0, 1] range")
        return image

    @staticmethod
    def _request_task(request: RawRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        observation = request.observation
        value = observation.get(
            "instruction",
            observation.get("prompt", ""),
        )
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ModelPackageError("OpenVLA-OFT prompt/instruction must be a string")
        return value

    def preprocess_one(self, request: RawRequest) -> dict[str, torch.Tensor]:
        """Return model inputs with an explicit leading batch dimension of one."""

        image = self._request_image(request)
        task = self._request_task(request)
        input_ids, attention_mask = self._tokenize(task)
        return {
            "input_ids": input_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "pixel_values": self._transform_image(image).unsqueeze(0),
        }


__all__ = [
    "OpenVLAOFTProcessor",
    "PROMPT_TEMPLATE",
    "SPACE_TOKEN",
]
