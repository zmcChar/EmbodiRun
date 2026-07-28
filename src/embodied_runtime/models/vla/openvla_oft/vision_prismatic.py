"""Prismatic DINOv2/SigLIP vision tower used by OpenVLA-OFT.

Adapted from ``vvla/policies/openvla_oft/vision_prismatic.py`` at vvla
commit ``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT,
Copyright 2026 Longxmas).  That file is in turn a port of the OpenVLA-OFT
checkpoint's ``modeling_prismatic.py`` leaf (Apache-2.0).

This module deliberately remains a model implementation detail.  Hardware
providers may later replace/export it, but the reference graph does not import
any backend-specific operator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import timm
import torch
from timm.models.vision_transformer import LayerScale


def _unpack_single(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return value[0]
    return value


def _single_intermediate(
    function: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return _unpack_single(function(*args, **kwargs))

    return wrapped


def _layer_scale_forward(self: LayerScale, value: torch.Tensor) -> torch.Tensor:
    return value.mul_(self.scale_factor) if self.inplace else value * self.scale_factor


def _patch_layer_scale(module: LayerScale) -> None:
    """Match the checkpoint's ``scale_factor`` parameter spelling."""

    module.scale_factor = torch.nn.Parameter(module.gamma.detach().clone())
    module.forward = _layer_scale_forward.__get__(module, LayerScale)
    del module.gamma


class PrismaticVisionBackbone(torch.nn.Module):
    """One or two fused timm ViTs returning second-to-last-block patches."""

    def __init__(
        self,
        *,
        use_fused_vision_backbone: bool,
        image_sizes: Sequence[int],
        timm_model_ids: Sequence[str],
        timm_override_act_layers: Sequence[str | None],
    ) -> None:
        super().__init__()
        model_ids = tuple(timm_model_ids)
        sizes = tuple(int(size) for size in image_sizes)
        activations = tuple(timm_override_act_layers)
        expected = 2 if use_fused_vision_backbone else 1
        if len(model_ids) != expected:
            raise ValueError(
                "fused Prismatic vision requires exactly two timm models; "
                "non-fused vision requires exactly one"
            )
        if len(sizes) < expected or len(activations) < expected:
            raise ValueError("each Prismatic vision tower needs an image size and activation")

        self.use_fused_vision_backbone = bool(use_fused_vision_backbone)
        self.num_images_in_input = 1
        self.featurizer = self._create_featurizer(
            model_ids[0],
            image_size=sizes[0],
            activation=activations[0],
        )
        self.embed_dim = int(self.featurizer.embed_dim)
        if self.use_fused_vision_backbone:
            self.fused_featurizer = self._create_featurizer(
                model_ids[1],
                image_size=sizes[1],
                activation=activations[1],
            )
            self.embed_dim += int(self.fused_featurizer.embed_dim)
        self._patch_layer_scales()

    @staticmethod
    def _create_featurizer(
        model_id: str,
        *,
        image_size: int,
        activation: str | None,
    ) -> torch.nn.Module:
        model = timm.create_model(
            model_id,
            pretrained=False,
            num_classes=0,
            img_size=image_size,
            act_layer=activation,
        )
        block_count = len(model.blocks)
        model.forward = _single_intermediate(
            partial(
                model.get_intermediate_layers,
                n={block_count - 2},
            )
        )
        return model

    def _patch_layer_scales(self) -> None:
        towers = [self.featurizer]
        if self.use_fused_vision_backbone:
            towers.append(self.fused_featurizer)
        for tower in towers:
            for module in tower.modules():
                if isinstance(module, LayerScale):
                    _patch_layer_scale(module)

    def get_num_patches(self) -> int:
        return int(self.featurizer.patch_embed.num_patches)

    def get_num_images_in_input(self) -> int:
        return self.num_images_in_input

    def set_num_images_in_input(self, count: int) -> None:
        if count <= 0:
            raise ValueError("num_images_in_input must be greater than zero")
        if count > 1 and not self.use_fused_vision_backbone:
            raise ValueError("multi-image OpenVLA-OFT requires a fused vision backbone")
        self.num_images_in_input = int(count)

    def _one_image(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            if pixels.shape[1] != 3:
                raise ValueError("non-fused Prismatic pixels require three channels")
            return self.featurizer(pixels)
        if pixels.shape[1] != 6:
            raise ValueError("fused Prismatic pixels require six channels per image")
        primary, fused = torch.split(pixels, (3, 3), dim=1)
        return torch.cat(
            (self.featurizer(primary), self.fused_featurizer(fused)),
            dim=-1,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        expected_channels = self.num_images_in_input * (6 if self.use_fused_vision_backbone else 3)
        if pixel_values.ndim != 4 or pixel_values.shape[1] != expected_channels:
            raise ValueError(
                "Prismatic pixel_values must have shape "
                f"[batch, {expected_channels}, height, width]"
            )
        if self.num_images_in_input == 1:
            return self._one_image(pixel_values)
        images = torch.split(pixel_values, 6, dim=1)
        return torch.cat(tuple(self._one_image(image) for image in images), dim=1)


class PrismaticProjector(torch.nn.Module):
    """Project Prismatic patch features into the Llama hidden dimension."""

    def __init__(
        self,
        *,
        use_fused_vision_backbone: bool,
        vision_dim: int,
        llm_dim: int,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = bool(use_fused_vision_backbone)
        self.vision_dim = int(vision_dim)
        self.llm_dim = int(llm_dim)
        if self.use_fused_vision_backbone:
            projection_dim = 4 * self.vision_dim
            self.fc1 = torch.nn.Linear(self.vision_dim, projection_dim)
            self.fc2 = torch.nn.Linear(projection_dim, self.llm_dim)
            self.fc3 = torch.nn.Linear(self.llm_dim, self.llm_dim)
            self.act_fn1 = torch.nn.GELU()
            self.act_fn2 = torch.nn.GELU()
        else:
            self.fc1 = torch.nn.Linear(self.vision_dim, self.llm_dim)
            self.fc2 = torch.nn.Linear(self.llm_dim, self.llm_dim)
            self.act_fn1 = torch.nn.GELU()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        hidden = self.act_fn1(self.fc1(patches))
        hidden = self.fc2(hidden)
        if self.use_fused_vision_backbone:
            hidden = self.fc3(self.act_fn2(hidden))
        return hidden


__all__ = ["PrismaticProjector", "PrismaticVisionBackbone"]
