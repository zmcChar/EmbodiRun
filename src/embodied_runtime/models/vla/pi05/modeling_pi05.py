"""Backend-independent Torch reference stages for pi0.5.

The stage split and most of the forward decomposition were refactored from
``vvla`` commit ``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT,
Copyright 2026 Longxmas).  The underlying pi0.5 PyTorch model is a port of
Physical Intelligence's OpenPI implementation maintained by Hugging Face
LeRobot under Apache-2.0:

* https://github.com/Physical-Intelligence/openpi
* https://github.com/huggingface/lerobot

This file intentionally contains only model semantics.  It does not select an
attention backend, capture CUDA graphs, create streams, or own static device
memory.  A hardware backend may compile/replace the ordinary Torch operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first = x[..., : x.shape[-1] // 2]
    second = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor | None,
) -> torch.Tensor:
    return residual + update if gate is None else residual + update * gate


def _rmsnorm(norm: Any, x: torch.Tensor, condition: torch.Tensor | None):
    """PiGemma RMSNorm semantics, including optional adaRMS modulation."""

    variance = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + norm.eps)
    if condition is None or norm.dense is None:
        return (normalized * (1.0 + norm.weight.float())).type_as(x), None
    modulation = norm.dense(condition)
    if x.ndim == 3:
        modulation = modulation.unsqueeze(1)
    scale, shift, gate = modulation.chunk(3, dim=-1)
    normalized = normalized * (1 + scale.float()) + shift.float()
    return normalized.to(x.dtype), gate.to(x.dtype)


def _mlp(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    return mlp.down_proj(mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))


def _repeat_kv(x: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return x
    batch, kv_heads, sequence, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, kv_heads, repetitions, sequence, head_dim)
        .reshape(batch, kv_heads * repetitions, sequence, head_dim)
    )


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Plain Torch attention matching Hugging Face Gemma eager semantics.

    This is intentionally not an ``AttentionBackend``.  It is the numerical
    reference that a Group-4 implementation must match when replacing it with
    SDPA, TensorRT, CANN, BPU kernels, or another provider.
    """

    repetitions = query.shape[1] // key.shape[1]
    key = _repeat_kv(key, repetitions)
    value = _repeat_kv(value, repetitions)
    if mask is not None:
        mask = mask[..., : key.shape[-2]]
        if mask.is_floating_point() and mask.dtype != query.dtype:
            mask = mask.to(query.dtype)
    attention = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if mask is not None:
        attention = attention + mask
    attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(attention, value)


class Pi05ReferenceModule(torch.nn.Module):
    """Four eager/reference entrypoints over one loaded LeRobot PI05Policy.

    Every public entrypoint accepts one mapping because
    :class:`embodied_runtime.contracts.BackendSession` submits a single ``TensorTree``.
    ``denoise_step`` returns only the velocity field.  Applying
    ``state += dt * velocity`` belongs to the generic execution engine.
    """

    def __init__(self, lerobot_policy: torch.nn.Module) -> None:
        super().__init__()
        # Keep one registered owner of all weights.  Accessing nested modules via
        # properties avoids registering the same tower repeatedly under aliases.
        self.lerobot_policy = lerobot_policy.eval()
        try:
            from lerobot.policies.pi05.modeling_pi05 import (
                create_sinusoidal_pos_embedding,
                make_att_2d_masks,
            )
        except ImportError as exc:  # pragma: no cover - guarded by adapter loading
            raise ImportError(
                "pi0.5 requires lerobot==0.5.1 and its transformers dependencies"
            ) from exc
        self._make_att_2d_masks = make_att_2d_masks
        self._sinusoidal = create_sinusoidal_pos_embedding

    @property
    def config(self):
        return self.lerobot_policy.config

    @property
    def _model(self):
        return self.lerobot_policy.model

    @property
    def _prefix_tower(self):
        return self._model.paligemma_with_expert.paligemma.model.language_model

    @property
    def _expert_tower(self):
        return self._model.paligemma_with_expert.gemma_expert.model

    def _attention_sublayer(
        self,
        attention: Any,
        hidden: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        mask: torch.Tensor,
        prefix_kv: tuple[torch.Tensor, torch.Tensor] | None,
        collected: list[tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> torch.Tensor:
        batch, sequence = hidden.shape[:2]
        head_dim = attention.head_dim
        query = attention.q_proj(hidden).view(batch, sequence, -1, head_dim).transpose(1, 2)
        key = attention.k_proj(hidden).view(batch, sequence, -1, head_dim).transpose(1, 2)
        value = attention.v_proj(hidden).view(batch, sequence, -1, head_dim).transpose(1, 2)
        query, key = _apply_rope(query, key, cosine, sine)
        if collected is not None:
            collected.append((key, value))
        if prefix_kv is not None:
            prefix_key, prefix_value = prefix_kv
            key = torch.cat((prefix_key, key), dim=2)
            value = torch.cat((prefix_value, value), dim=2)
        output = _reference_attention(
            query,
            key,
            value,
            mask=mask,
            scaling=attention.scaling,
        )
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        return attention.o_proj(output)

    def _tower_forward(
        self,
        tower: Any,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        adarms_condition: torch.Tensor | None,
        *,
        prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        collect: bool = False,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...] | None]:
        hidden = hidden.to(tower.layers[0].self_attn.q_proj.weight.dtype)
        cosine, sine = tower.rotary_emb(hidden, position_ids)
        collected: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if collect else None
        for index, layer in enumerate(tower.layers):
            residual = hidden
            normalized, gate = _rmsnorm(
                layer.input_layernorm,
                hidden,
                adarms_condition,
            )
            update = self._attention_sublayer(
                layer.self_attn,
                normalized,
                cosine,
                sine,
                mask,
                None if prefix_kv is None else prefix_kv[index],
                collected,
            )
            hidden = _gated_residual(residual, update, gate)
            residual = hidden
            normalized, gate = _rmsnorm(
                layer.post_attention_layernorm,
                hidden,
                adarms_condition,
            )
            hidden = _gated_residual(residual, _mlp(layer.mlp, normalized), gate)
        hidden, _ = _rmsnorm(tower.norm, hidden, adarms_condition)
        return hidden, None if collected is None else tuple(collected)

    def encode_prefix(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Encode images and language once and return read-only per-layer K/V."""

        model = self._model
        prefix_embeddings, prefix_pad_masks, prefix_attention_masks = model.embed_prefix(
            inputs["images"],
            inputs["image_masks"],
            inputs["tokens"],
            inputs["token_masks"],
        )
        attention_2d = self._make_att_2d_masks(
            prefix_pad_masks,
            prefix_attention_masks,
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mask_4d = model._prepare_attention_masks_4d(attention_2d)
        _, kv = self._tower_forward(
            self._prefix_tower,
            prefix_embeddings,
            position_ids,
            mask_4d,
            None,
            collect=True,
        )
        return {
            "kv": kv,
            "prefix_pad_masks": prefix_pad_masks,
        }

    def init_state(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Create initial action noise, preferring explicit noise over RNG inputs."""

        weight = self._model.action_in_proj.weight
        batch_size = int(inputs["batch_size"])
        shape = (
            batch_size,
            int(self.config.chunk_size),
            int(self.config.max_action_dim),
        )
        noise = inputs.get("noise")
        if noise is not None:
            tensor = torch.as_tensor(noise, device=weight.device, dtype=weight.dtype)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"pi0.5 noise has shape {tuple(tensor.shape)}, expected {shape}")
            return tensor
        generator = inputs.get("generator")
        if generator is None and inputs.get("seed") is not None:
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(int(inputs["seed"]))
        return torch.randn(
            shape,
            device=weight.device,
            dtype=weight.dtype,
            generator=generator,
        )

    def _embed_suffix(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        model = self._model
        time_embedding = self._sinusoidal(
            time,
            model.action_in_proj.out_features,
            min_period=model.config.min_period,
            max_period=model.config.max_period,
            device=time.device,
        ).type(dtype=time.dtype)
        action_embedding = model.action_in_proj(state)
        condition = F.silu(model.time_mlp_in(time_embedding))
        condition = F.silu(model.time_mlp_out(condition))
        batch, suffix_length = action_embedding.shape[:2]
        pad_masks = torch.ones(
            batch,
            suffix_length,
            dtype=torch.bool,
            device=state.device,
        )
        # pi0.5 action tokens form one bidirectional block: [1, 0, ..., 0].
        attention_masks = torch.zeros(
            batch,
            suffix_length,
            dtype=action_embedding.dtype,
            device=state.device,
        )
        attention_masks[:, 0] = 1
        return action_embedding, pad_masks, attention_masks, condition

    def denoise_step(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Return ``v(state, time | prefix)`` without integrating the state."""

        state = inputs["state"]
        prefix = inputs["prefix"]
        time = inputs["time"]
        if not isinstance(time, torch.Tensor):
            time = torch.full(
                (state.shape[0],),
                float(time),
                device=state.device,
                dtype=state.dtype,
            )
        elif time.ndim == 0:
            time = time.to(device=state.device, dtype=state.dtype).expand(state.shape[0])
        else:
            time = time.to(device=state.device, dtype=state.dtype)

        suffix_embeddings, suffix_pad_masks, suffix_attention_masks, condition = self._embed_suffix(
            state, time
        )
        prefix_pad_masks = prefix["prefix_pad_masks"]
        batch_size, prefix_length = prefix_pad_masks.shape
        suffix_length = suffix_pad_masks.shape[1]
        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_length,
            prefix_length,
        )
        suffix_attention_2d = self._make_att_2d_masks(
            suffix_pad_masks,
            suffix_attention_masks,
        )
        full_attention_2d = torch.cat(
            (prefix_pad_2d, suffix_attention_2d),
            dim=2,
        )
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        mask_4d = self._model._prepare_attention_masks_4d(full_attention_2d)
        hidden, _ = self._tower_forward(
            self._expert_tower,
            suffix_embeddings,
            position_ids,
            mask_4d,
            condition,
            prefix_kv=prefix["kv"],
        )
        hidden = hidden[:, -self.config.chunk_size :].to(self._model.action_out_proj.weight.dtype)
        return self._model.action_out_proj(hidden)

    @staticmethod
    def finalize(inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Expose the integrated flow state as the predicted action chunk."""

        return {"actions": inputs["state"]}


__all__ = ["Pi05ReferenceModule"]
