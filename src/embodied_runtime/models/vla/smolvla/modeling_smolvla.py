"""Backend-neutral staged inference over a loaded LeRobot SmolVLA policy."""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import Any

import torch

from ...plans.iterative_flow import IterativeFlowPlan

ACTION = "action"
LANGUAGE_TOKENS = "observation.language.tokens"
LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def _make_attention_masks(
    pad_masks: torch.Tensor,
    attention_masks: torch.Tensor,
) -> torch.Tensor:
    """Match LeRobot 0.3.3's prefix/block attention construction."""

    if pad_masks.ndim != 2 or attention_masks.ndim != 2:
        raise ValueError("SmolVLA pad and attention masks must be two-dimensional")
    cumulative = torch.cumsum(attention_masks, dim=1)
    causal_blocks = cumulative[:, None, :] <= cumulative[:, :, None]
    valid_tokens = pad_masks[:, None, :] * pad_masks[:, :, None]
    return causal_blocks & valid_tokens


class SmolVLAFlowPlan(IterativeFlowPlan):
    """Reproduce LeRobot 0.3.3's scalar FP32 time accumulation exactly."""

    def schedule(self, num_steps: int | None = None) -> tuple[tuple[float, float], ...]:
        steps = self.default_num_steps if num_steps is None else num_steps
        if steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        dt = _round_float32(-1.0 / steps)
        time = _round_float32(1.0)
        schedule = []
        for _ in range(steps):
            schedule.append((time, dt))
            time = _round_float32(time + dt)
        return tuple(schedule)


def _round_float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


class SmolVLAReferenceModule(torch.nn.Module):
    """Expose SmolVLA's prefix, flow step, and output stages to the engine.

    LeRobot owns the neural-network semantics. The generic engine owns the
    Euler loop, safe points, cancellation, and scheduling, while a backend owns
    placement and execution of each ordinary Torch entrypoint.
    """

    def __init__(self, lerobot_policy: torch.nn.Module) -> None:
        super().__init__()
        self.lerobot_policy = lerobot_policy.eval()

    @property
    def config(self) -> Any:
        return self.lerobot_policy.config

    @property
    def _model(self) -> Any:
        return self.lerobot_policy.model

    @property
    def native_action_dim(self) -> int:
        action_feature = getattr(self.config, "action_feature", None)
        if action_feature is not None:
            return int(action_feature.shape[0])
        output_features = getattr(self.config, "output_features", None) or {}
        action_feature = output_features.get(ACTION)
        if action_feature is not None:
            return int(action_feature.shape[0])
        return int(self.config.max_action_dim)

    def encode_prefix(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Normalize observations and build the reusable image/language KV cache."""

        batch = dict(inputs)
        prepare_batch = getattr(self.lerobot_policy, "_prepare_batch", None)
        if callable(prepare_batch):
            batch = prepare_batch(batch)

        images, image_masks = self.lerobot_policy.prepare_images(batch)
        state = self.lerobot_policy.prepare_state(batch)
        try:
            language_tokens = batch[LANGUAGE_TOKENS]
            language_masks = batch[LANGUAGE_ATTENTION_MASK]
        except KeyError as error:
            raise ValueError(f"SmolVLA input is missing {error.args[0]!r}") from error

        prefix_embeddings, prefix_pad_masks, prefix_attention_masks = self._model.embed_prefix(
            images,
            image_masks,
            language_tokens,
            language_masks,
            state=state,
        )
        prefix_attention_2d = _make_attention_masks(
            prefix_pad_masks,
            prefix_attention_masks,
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self._model.vlm_with_expert.forward(
            attention_mask=prefix_attention_2d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embeddings, None],
            use_cache=bool(self.config.use_cache),
            fill_kv_cache=True,
        )
        return {
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def init_state(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Create the FP32 action noise used as flow state."""

        weight = self._model.action_in_proj.weight
        batch_size = int(inputs["batch_size"])
        shape = (
            batch_size,
            int(self.config.chunk_size),
            int(self.config.max_action_dim),
        )
        noise = inputs.get("noise")
        if noise is not None:
            tensor = torch.as_tensor(noise, dtype=torch.float32, device=weight.device)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"SmolVLA noise has shape {tuple(tensor.shape)}, expected {shape}")
            return tensor

        generator = inputs.get("generator")
        if generator is None and inputs.get("seed") is not None:
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(int(inputs["seed"]))
        return torch.randn(
            shape,
            dtype=torch.float32,
            device=weight.device,
            generator=generator,
        )

    def denoise_step(self, inputs: Mapping[str, Any]) -> torch.Tensor:
        """Return one SmolVLA velocity field without applying the Euler update."""

        state = inputs["state"]
        prefix = inputs["prefix"]
        time = inputs["time"]
        if not isinstance(state, torch.Tensor):
            raise TypeError("SmolVLA flow state must be a torch.Tensor")
        if not isinstance(time, torch.Tensor):
            timestep = torch.full(
                (state.shape[0],),
                float(time),
                dtype=torch.float32,
                device=state.device,
            )
        elif time.ndim == 0:
            timestep = time.to(device=state.device, dtype=torch.float32).expand(state.shape[0])
        else:
            timestep = time.to(device=state.device, dtype=torch.float32)

        return self._model.denoise_step(
            prefix_pad_masks=prefix["prefix_pad_masks"],
            past_key_values=prefix["past_key_values"],
            x_t=state,
            timestep=timestep,
        )

    def finalize(self, inputs: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        """Crop padding and restore the checkpoint's physical action scale."""

        actions = inputs["state"][..., : self.native_action_dim]
        unnormalize = getattr(self.lerobot_policy, "unnormalize_outputs", None)
        if callable(unnormalize):
            actions = unnormalize({ACTION: actions})[ACTION]
        if bool(getattr(self.config, "adapt_to_pi_aloha", False)):
            actions = self.lerobot_policy._pi_aloha_encode_actions(actions)
        return {ACTION + "s": actions}


__all__ = [
    "LANGUAGE_ATTENTION_MASK",
    "LANGUAGE_TOKENS",
    "SmolVLAFlowPlan",
    "SmolVLAReferenceModule",
]
