"""Reference single-forward graph for OpenVLA-OFT.

Adapted from vvla commit ``80b5cf48c8710c69ed97200903562e9787efe105``
(MIT) and checked against RLinf's Apache-2.0
``OpenVLAOFTForRLActionPrediction`` execution semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .head import CategoricalActionHead


class OpenVLAOFTReferenceModule(torch.nn.Module):
    """Full causal forward from processed observation to one action chunk."""

    def __init__(
        self,
        vision_backbone: torch.nn.Module,
        projector: torch.nn.Module,
        language_model: torch.nn.Module,
        head: CategoricalActionHead,
    ) -> None:
        super().__init__()
        self.vision_backbone = vision_backbone
        self.projector = projector
        self.language_model = language_model
        self.head = head
        self.action_dim = head.action_dim
        self.action_horizon = head.num_action_chunks
        self.n_tokens = head.n_tokens

    def _input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        getter = getattr(self.language_model, "get_input_embeddings", None)
        if callable(getter):
            embeddings = getter()
        else:
            model = getattr(self.language_model, "model", None)
            embeddings = getattr(model, "embed_tokens", None)
        if embeddings is None:
            raise TypeError(
                "OpenVLA-OFT language model must expose get_input_embeddings() "
                "or model.embed_tokens"
            )
        return embeddings(input_ids)

    def forward(self, inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Run one deterministic OpenVLA-OFT action prediction."""

        try:
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            pixel_values = inputs["pixel_values"]
        except KeyError as error:
            raise ValueError(f"OpenVLA-OFT input is missing {error.args[0]!r}") from error
        if not all(
            isinstance(value, torch.Tensor) for value in (input_ids, attention_mask, pixel_values)
        ):
            raise TypeError("OpenVLA-OFT inputs must be torch.Tensor leaves")
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have matching [batch, length]")
        if pixel_values.ndim != 4 or pixel_values.shape[0] != input_ids.shape[0]:
            raise ValueError("pixel_values must have shape [batch, channels, height, width]")
        if input_ids.shape[1] < 1:
            raise ValueError("OpenVLA-OFT prompt must contain at least one token")
        if torch.any(attention_mask[:, -1] == 0):
            raise ValueError("the final OpenVLA-OFT prompt token must be active")

        batch_size = input_ids.shape[0]
        placeholders = torch.ones(
            (batch_size, self.n_tokens),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        complete_ids = torch.cat((input_ids, placeholders), dim=1)
        action_mask = torch.ones(
            (batch_size, self.n_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        complete_mask = torch.cat((attention_mask, action_mask), dim=1)

        token_embeddings = self._input_embeddings(complete_ids)
        # Placeholder ids allocate causal positions; the action queries carry
        # no token content in the native OpenVLA-OFT implementation.
        token_embeddings = token_embeddings.clone()
        token_embeddings[:, -self.n_tokens :] = 0

        vision_dtype = _floating_dtype(self.vision_backbone)
        if vision_dtype is not None and pixel_values.dtype != vision_dtype:
            pixel_values = pixel_values.to(dtype=vision_dtype)
        vision_features = self.vision_backbone(pixel_values)
        projector_dtype = _floating_dtype(self.projector)
        if projector_dtype is not None and vision_features.dtype != projector_dtype:
            vision_features = vision_features.to(dtype=projector_dtype)
        patch_embeddings = self.projector(vision_features)
        if patch_embeddings.dtype != token_embeddings.dtype:
            patch_embeddings = patch_embeddings.to(dtype=token_embeddings.dtype)
        if patch_embeddings.ndim != 3:
            raise ValueError("OpenVLA-OFT projected vision features must have shape [B,P,D]")
        if (
            patch_embeddings.shape[0] != batch_size
            or patch_embeddings.shape[-1] != token_embeddings.shape[-1]
        ):
            raise ValueError(
                "OpenVLA-OFT vision patches do not match language batch/hidden dimensions"
            )

        multimodal_embeddings = torch.cat(
            (
                token_embeddings[:, :1],
                patch_embeddings,
                token_embeddings[:, 1:],
            ),
            dim=1,
        )
        patch_mask = torch.ones(
            patch_embeddings.shape[:2],
            dtype=complete_mask.dtype,
            device=complete_mask.device,
        )
        multimodal_mask = torch.cat(
            (
                complete_mask[:, :1],
                patch_mask,
                complete_mask[:, 1:],
            ),
            dim=1,
        )
        position_ids = multimodal_mask.long().cumsum(dim=1) - 1
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=multimodal_embeddings,
            attention_mask=multimodal_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = _output_logits(outputs)
        expected_sequence = multimodal_embeddings.shape[1]
        if logits.ndim != 3 or logits.shape[:2] != (
            batch_size,
            expected_sequence,
        ):
            raise ValueError(
                "OpenVLA-OFT language logits must have shape "
                f"[{batch_size}, {expected_sequence}, vocabulary]"
            )

        # Prompt space predicts action 0; action-query i predicts action i+1.
        # The final query predicts STOP and is deliberately not consumed.
        action_logits = logits[:, -self.n_tokens - 1 : -1]
        action_tokens, _ = self.head.greedy(action_logits)
        actions = self.head.tokens_to_actions(action_tokens)
        return {
            "actions": actions,
            "action_tokens": action_tokens,
        }


def _output_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    raise TypeError("OpenVLA-OFT language model output must expose 'logits'")


def _floating_dtype(module: torch.nn.Module) -> torch.dtype | None:
    for value in (*module.parameters(), *module.buffers()):
        if value.is_floating_point():
            return value.dtype
    return None


__all__ = ["OpenVLAOFTReferenceModule"]
