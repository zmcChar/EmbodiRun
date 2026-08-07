"""Lazy policy-runtime helpers shared by VLABench applications."""

from __future__ import annotations

from typing import Any


def seed_policy(seed: int) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the VLABench policy environment requires PyTorch") from error
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_smolvla_runner(*args: Any, **kwargs: Any) -> Any:
    from embodied_runtime.integrations.lerobot.vlabench_smolvla import VLABenchSmolVLARunner

    return VLABenchSmolVLARunner(*args, **kwargs)


__all__ = ["make_smolvla_runner", "seed_policy"]
