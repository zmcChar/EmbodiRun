from __future__ import annotations

import asyncio
from functools import wraps

import numpy as np
import pytest

from embodied_runtime.contracts import InferenceRequest, InferenceResult
from embodied_runtime.integrations.serving.gr00t import HfLocalGr00tProvider
from embodied_runtime.models.vla.gr00t_n17 import (
    Gr00tN17Adapter,
    synthetic_droid_request,
)


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


class _FakeLocalEngine:
    def __init__(self) -> None:
        self.requests: list[InferenceRequest] = []
        self.closed = False

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        self.requests.append(request)
        return InferenceResult(
            request_id=request.request_id,
            output={
                "actions": {
                    "eef_9d": np.zeros((40, 9), dtype=np.float32),
                    "gripper_position": np.zeros((40, 1), dtype=np.float32),
                    "joint_position": np.zeros((40, 7), dtype=np.float32),
                },
                "info": {},
            },
            metadata={"backend": "torch_cuda", "device_id": "cuda:0"},
        )

    async def aclose(self) -> None:
        self.closed = True


@async_test
async def test_hf_provider_preprocesses_raw_request_and_preserves_engine_metadata() -> None:
    adapter = Gr00tN17Adapter()
    engine = _FakeLocalEngine()
    provider = HfLocalGr00tProvider(adapter, engine)  # type: ignore[arg-type]

    result = await provider.infer_async(
        InferenceRequest(
            request_id="hf-request",
            payload=synthetic_droid_request(image_height=8, image_width=8),
        )
    )
    chunk = adapter.postprocess_one(result.output)

    assert result.metadata["provider"] == "hf"
    assert result.metadata["provider_runtime"] == "nvidia_isaac_gr00t"
    assert result.metadata["backend"] == "torch_cuda"
    assert engine.requests[0].payload["video"]["exterior_image_1_left"].shape == (
        1,
        2,
        8,
        8,
        3,
    )
    assert chunk.actions["eef_9d"].shape == (40, 9)

    await provider.aclose()
    assert engine.closed


def test_hf_provider_rejects_full_policy_compile_boundary() -> None:
    with pytest.raises(ValueError, match="only mode='eager'"):
        HfLocalGr00tProvider.from_checkpoint("unused", mode="compile")
