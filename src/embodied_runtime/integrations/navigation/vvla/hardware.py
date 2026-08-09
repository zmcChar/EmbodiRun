"""CUDA capacity guards for the pinned ActiveVLN checkpoint."""

from __future__ import annotations

from typing import Any

from embodied_runtime.policies.navigation.errors import NavigationPolicyError

# The fp32 checkpoint weights alone occupy about 15.15 GiB. A lower bound of
# 20 GiB leaves only modest room for CUDA context, activations and KV state;
# cards below it must use a 16-bit dtype instead of failing halfway through load.
MIN_ACTIVEVLN_FP32_CUDA_BYTES = 20 * 2**30


def require_activevln_cuda_capacity(torch: Any, device: str, dtype: str) -> None:
    """Reject a predictably impossible fp32/auto load before allocating weights."""

    if dtype not in {"auto", "float32"} or not str(device).startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return
    total_memory = int(torch.cuda.get_device_properties(device).total_memory)
    if total_memory < MIN_ACTIVEVLN_FP32_CUDA_BYTES:
        total_gib = total_memory / 2**30
        raise NavigationPolicyError(
            "ActiveVLN dtype="
            f"{dtype!r} requires fp32 weights and cannot fit safely on this "
            f"{total_gib:.1f} GiB GPU; select float16 or bfloat16"
        )


__all__ = ["MIN_ACTIVEVLN_FP32_CUDA_BYTES", "require_activevln_cuda_capacity"]
