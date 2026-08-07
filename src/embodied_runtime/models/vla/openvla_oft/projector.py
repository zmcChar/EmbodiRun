"""Prismatic patch projector used by OpenVLA-OFT."""

from __future__ import annotations

import torch


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


__all__ = ["PrismaticProjector"]
