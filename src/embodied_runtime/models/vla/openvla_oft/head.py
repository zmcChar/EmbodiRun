"""Discrete action-token head for the RLinf OpenVLA-OFT variant.

Adapted from ``vvla/policies/openvla_oft/head.py`` at vvla commit
``80b5cf48c8710c69ed97200903562e9787efe105`` (MIT,
Copyright 2026 Longxmas).

Only the top ``n_action_bins`` entries of the unpadded text vocabulary represent
actions. Each action position is sampled independently, then converted from an
absolute token id into a normalized bin center and, where requested, back to the
dataset action scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F


class CategoricalActionHead(torch.nn.Module):
    """Sample, score, and decode OpenVLA-OFT categorical action tokens."""

    def __init__(
        self,
        vocab_size: int,
        n_action_bins: int,
        action_dim: int,
        num_action_chunks: int,
        q01: Sequence[float] | Any,
        q99: Sequence[float] | Any,
        mask: Sequence[bool] | Any,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n_action_bins = int(n_action_bins)
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero")
        if not 1 < self.n_action_bins <= self.vocab_size:
            raise ValueError("n_action_bins must be in [2, vocab_size]")
        if self.action_dim <= 0 or self.num_action_chunks <= 0:
            raise ValueError("action_dim and num_action_chunks must be greater than zero")
        self.n_tokens = self.action_dim * self.num_action_chunks

        # Match vvla/RLinf: make N evenly-spaced endpoints, then use the N-1
        # interval centers. Float64 construction followed by float32 storage
        # preserves NumPy's original numerical convention without requiring
        # NumPy at runtime.
        bins = torch.linspace(-1.0, 1.0, self.n_action_bins, dtype=torch.float64)
        bin_centers = ((bins[:-1] + bins[1:]) / 2.0).to(torch.float32)
        q01_tensor = torch.as_tensor(q01, dtype=torch.float32).flatten()
        q99_tensor = torch.as_tensor(q99, dtype=torch.float32).flatten()
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).flatten()
        expected = (self.action_dim,)
        for name, value in (
            ("q01", q01_tensor),
            ("q99", q99_tensor),
            ("mask", mask_tensor),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must contain one value per action dimension; "
                    f"expected {expected}, got {tuple(value.shape)}"
                )
        if torch.any(q99_tensor < q01_tensor):
            raise ValueError("q99 must be greater than or equal to q01 elementwise")

        self.register_buffer("bin_centers", bin_centers, persistent=False)
        self.register_buffer("q01", q01_tensor, persistent=False)
        self.register_buffer("q99", q99_tensor, persistent=False)
        self.register_buffer("norm_mask", mask_tensor, persistent=False)

    def _validate_logits(self, logits: torch.Tensor) -> None:
        if logits.ndim != 3:
            raise ValueError(
                "OpenVLA-OFT logits must have shape [batch, action_tokens, vocabulary]"
            )
        if logits.shape[1] != self.n_tokens:
            raise ValueError(
                f"expected {self.n_tokens} action-token positions, got {logits.shape[1]}"
            )
        if logits.shape[-1] < self.vocab_size:
            raise ValueError(
                f"logits vocabulary has size {logits.shape[-1]}, "
                f"smaller than configured vocab_size={self.vocab_size}"
            )

    def _validate_token_ids(self, idxs: torch.Tensor) -> None:
        if idxs.ndim != 2 or idxs.shape[1] != self.n_tokens:
            raise ValueError(f"action token ids must have shape [batch, {self.n_tokens}]")

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask text tokens and any padded vocabulary entries."""

        self._validate_logits(logits)
        masked = logits.clone()
        first_action_token = self.vocab_size - self.n_action_bins
        masked[..., :first_action_token] = -torch.inf
        masked[..., self.vocab_size :] = -torch.inf
        return masked

    @staticmethod
    def _top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            return logits
        k = min(top_k, logits.shape[-1])
        threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
        return logits.masked_fill(logits < threshold, -torch.inf)

    def greedy(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return greedy token ids and their per-token log-probabilities."""

        return self.sample(logits, do_sample=False)

    def sample(
        self,
        logits: torch.Tensor,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = -1,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return absolute token ids and behavior log-probabilities.

        Both outputs have shape ``[batch, action_dim * num_action_chunks]``.
        Greedy decoding deliberately ignores ``temperature`` and ``top_k``, as
        in the RLinf reference implementation.
        """

        processed = self._mask_logits(logits)
        if do_sample:
            if temperature <= 0:
                raise ValueError("temperature must be greater than zero when sampling")
            processed = self._top_k(processed / temperature, top_k)
            probabilities = F.softmax(processed, dim=-1)
            flat = probabilities.reshape(-1, probabilities.shape[-1])
            idxs = torch.multinomial(
                flat,
                num_samples=1,
                generator=generator,
            ).reshape(probabilities.shape[:-1])
        else:
            idxs = processed.argmax(dim=-1)
        return idxs, self._logprob_from_processed(processed, idxs)

    def logprob(
        self,
        logits: torch.Tensor,
        idxs: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = -1,
    ) -> torch.Tensor:
        """Differentiably score action tokens under the configured distribution."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self._validate_token_ids(idxs)
        processed = self._top_k(self._mask_logits(logits) / temperature, top_k)
        return self._logprob_from_processed(processed, idxs)

    def recompute_logprob(
        self,
        logits: torch.Tensor,
        idxs: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = -1,
    ) -> torch.Tensor:
        """Compatibility spelling used by vvla's rollout implementation."""

        return self.logprob(logits, idxs, temperature=temperature, top_k=top_k)

    def _logprob_from_processed(
        self,
        processed: torch.Tensor,
        idxs: torch.Tensor,
    ) -> torch.Tensor:
        # RLinf re-applies the action-bin mask before cross entropy. The second
        # mask is mathematically idempotent, but retaining it preserves the
        # reference operation order for parity work.
        action_logits = self._mask_logits(processed)
        vocab = action_logits.shape[-1]
        logprob = -F.cross_entropy(
            action_logits.reshape(-1, vocab),
            idxs.reshape(-1),
            reduction="none",
        )
        return logprob.view_as(idxs).float()

    @torch.no_grad()
    def tokens_to_actions(self, idxs: torch.Tensor) -> torch.Tensor:
        """Convert absolute token ids into an unnormalized action chunk."""

        self._validate_token_ids(idxs)
        discretized = self.vocab_size - idxs
        discretized = torch.clamp(
            discretized - 1,
            0,
            self.bin_centers.shape[0] - 1,
        )
        normalized = self.bin_centers[discretized]
        normalized = normalized.reshape(
            idxs.shape[0],
            self.num_action_chunks,
            self.action_dim,
        )
        unnormalized = 0.5 * (normalized + 1.0) * (self.q99 - self.q01 + 1e-8) + self.q01
        return torch.where(self.norm_mask, unnormalized, normalized)


__all__ = ["CategoricalActionHead"]
