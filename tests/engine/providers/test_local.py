from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from embodied_runtime.backends import (
    ArtifactVariant,
    BackendRegistry,
    CompileOptions,
    DeviceInfo,
    MemoryStats,
    SupportReport,
)
from embodied_runtime.engine import (
    ExecutionContext,
    InferenceRequest,
)
from embodied_runtime.engine.providers import LocalBackendProvider
from embodied_runtime.models.package import ModelPackage
from embodied_runtime.models.plans import SingleForwardPlan
from embodied_runtime.models.request import RawRequest
from embodied_runtime.models.spec import ModelSpec


class _Adapter:
    def __init__(self) -> None:
        self.preprocessed: list[RawRequest] = []
        self.package_options: dict[str, Any] | None = None

    def describe(self) -> ModelSpec:
        return ModelSpec(model_id="provider-fixture", family="test")

    def build_package(self, checkpoint: str, **options: Any) -> ModelPackage:
        self.package_options = {"checkpoint": checkpoint, **options}
        plan = SingleForwardPlan()
        return ModelPackage(
            spec=self.describe(),
            entrypoints={plan.forward: lambda payload: payload},
            plan=plan,
            checkpoint=checkpoint,
            package_id="provider-fixture-package",
        )

    def preprocess_one(self, request: RawRequest) -> dict[str, Any]:
        self.preprocessed.append(request)
        return {"value": request.observation["value"]}

    def collate(self, samples):
        return samples

    def unbatch(self, outputs, batch_size: int):
        assert batch_size == 1
        return [outputs]

    def postprocess_one(self, output):
        return output


class _BlockingAdapter(_Adapter):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def preprocess_one(self, request: RawRequest) -> dict[str, Any]:
        self.started.set()
        self.release.wait()
        return super().preprocess_one(request)


class _Session:
    def __init__(self, device: DeviceInfo, package_id: str) -> None:
        self._device = device
        self._package_id = package_id
        self.closed = False

    @property
    def package_id(self) -> str:
        return self._package_id

    @property
    def device(self) -> DeviceInfo:
        return self._device

    def submit(
        self,
        entrypoint: str,
        inputs: Any,
        context: ExecutionContext,
    ) -> Any:
        assert entrypoint == "forward"
        assert context.request_ids
        return inputs

    def add_scaled(self, state, update, scale, context):
        raise AssertionError("single-forward provider must not update flow state")

    def memory_stats(self) -> MemoryStats:
        return MemoryStats(
            allocated_bytes=0,
            reserved_bytes=0,
            total_bytes=1024,
            free_bytes=1024,
        )

    def close(self) -> None:
        self.closed = True


class _VendorBackend:
    name = "vendor_test"

    def __init__(self) -> None:
        self.device = DeviceInfo(
            backend=self.name,
            device_id="vendor:0",
            kind="accelerator",
            vendor="test-vendor",
            name="test accelerator",
            capabilities=frozenset({"vendor_runtime"}),
        )
        self.compile_options: CompileOptions | None = None
        self.session: _Session | None = None

    def probe(self) -> tuple[DeviceInfo, ...]:
        return (self.device,)

    def supports(self, package, device) -> SupportReport:
        if device == self.device and package.spec.model_id == "provider-fixture":
            return SupportReport.yes("vendor_runtime")
        return SupportReport.no("unsupported fixture")

    def compile(self, package, device, options) -> ArtifactVariant:
        self.compile_options = options
        return ArtifactVariant(
            backend=self.name,
            device=device,
            package=package,
            options=options,
            payload={"compiled": True},
        )

    def load(self, artifact) -> _Session:
        self.session = _Session(artifact.device, artifact.package.package_id)
        return self.session


def test_local_provider_runs_through_an_injected_non_cuda_backend() -> None:
    adapter = _Adapter()
    backend = _VendorBackend()
    registry = BackendRegistry([backend])
    options = CompileOptions(
        mode="vendor_compile",
        dtype="bf16",
        options={"graph": True},
    )
    provider = LocalBackendProvider.from_checkpoint(
        adapter,
        "fixture-checkpoint",
        backends=registry,
        backend_name="vendor_test",
        device="vendor:0",
        compile_options=options,
        package_options={"revision": "test-revision"},
        provider_name="hf",
        provider_runtime="hf_vendor_test",
    )

    async def run() -> None:
        request = InferenceRequest(
            request_id="provider-request",
            payload=RawRequest(observation={"value": 7}),
            priority=4,
            metadata={"source": "test"},
        )
        result = await provider.infer_async(request)
        assert result.request_id == "provider-request"
        assert result.output == {"value": 7}
        assert result.metadata["backend"] == "vendor_test"
        assert result.metadata["device_id"] == "vendor:0"
        assert result.metadata["provider"] == "hf"
        assert result.metadata["provider_runtime"] == "hf_vendor_test"
        await provider.aclose()
        await provider.aclose()

    asyncio.run(run())

    assert adapter.preprocessed
    assert adapter.package_options == {
        "checkpoint": "fixture-checkpoint",
        "revision": "test-revision",
    }
    assert backend.compile_options is options
    assert backend.session is not None and backend.session.closed
    assert provider.capabilities.device == backend.device
    assert "vendor_runtime" in provider.capabilities.features


def test_local_provider_reports_detected_devices_for_a_bad_selection() -> None:
    adapter = _Adapter()
    backend = _VendorBackend()
    registry = BackendRegistry([backend])

    with pytest.raises(RuntimeError, match="vendor_test/vendor:0"):
        LocalBackendProvider.from_checkpoint(
            adapter,
            "fixture-checkpoint",
            backends=registry,
            backend_name="vendor_test",
            device="vendor:9",
        )


def test_local_provider_close_drains_cancelled_preprocessing_thread() -> None:
    adapter = _BlockingAdapter()
    backend = _VendorBackend()
    provider = LocalBackendProvider.from_checkpoint(
        adapter,
        "fixture-checkpoint",
        backends=BackendRegistry([backend]),
        backend_name="vendor_test",
        device="vendor:0",
    )

    async def run() -> None:
        inference = asyncio.create_task(
            provider.infer_async(
                InferenceRequest(
                    request_id="cancelled-preprocess",
                    payload=RawRequest(observation={"value": 7}),
                )
            )
        )
        started = await asyncio.to_thread(adapter.started.wait, 1.0)
        assert started
        inference.cancel()

        closing = asyncio.create_task(provider.aclose())
        await asyncio.sleep(0)
        assert not inference.done()
        assert not closing.done()
        assert backend.session is not None and not backend.session.closed

        adapter.release.set()
        with pytest.raises(asyncio.CancelledError):
            await inference
        await closing
        assert backend.session.closed

    asyncio.run(run())
