from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodied_runtime.integrations.navigation.vvla.hardware import (
    MIN_ACTIVEVLN_FP32_CUDA_BYTES,
    require_activevln_cuda_capacity,
)
from embodied_runtime.policies.navigation.errors import NavigationPolicyError


def _torch_with_cuda_bytes(total_memory: int):
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda device: SimpleNamespace(total_memory=total_memory),
    )
    return SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize("dtype", ["auto", "float32"])
def test_fp32_profile_fails_before_loading_on_a_16_gib_gpu(dtype: str) -> None:
    torch = _torch_with_cuda_bytes(16 * 2**30)

    with pytest.raises(NavigationPolicyError, match="cannot fit safely.*bfloat16"):
        require_activevln_cuda_capacity(torch, "cuda:0", dtype)


def test_capacity_guard_allows_16_bit_or_a_large_fp32_gpu() -> None:
    small = _torch_with_cuda_bytes(16 * 2**30)
    require_activevln_cuda_capacity(small, "cuda:0", "bfloat16")

    large = _torch_with_cuda_bytes(MIN_ACTIVEVLN_FP32_CUDA_BYTES)
    require_activevln_cuda_capacity(large, "cuda:0", "float32")
