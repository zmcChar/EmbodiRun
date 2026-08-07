from __future__ import annotations

import pytest

from embodied_runtime.backends import (
    AscendBackend,
    BackendRegistry,
    DeviceInfo,
    HorizonBackend,
    TorchCudaBackend,
)
from embodied_runtime.models.package import ModelPackage
from embodied_runtime.models.plans import IterativeFlowPlan
from embodied_runtime.models.spec import ModelSpec


def _identity_package() -> ModelPackage:
    entrypoints = {
        name: (lambda payload: payload) for name in IterativeFlowPlan().required_entrypoints()
    }
    return ModelPackage(
        spec=ModelSpec(model_id="tiny", family="test"),
        entrypoints=entrypoints,
        plan=IterativeFlowPlan(),
    )


def test_registry_rejects_duplicate_backend_names() -> None:
    registry = BackendRegistry([TorchCudaBackend()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(TorchCudaBackend())


def test_registry_selects_local_cpu_reference_device() -> None:
    backend = TorchCudaBackend()
    if not backend.probe():
        pytest.skip("PyTorch is not installed")
    registry = BackendRegistry([backend])
    selected, device, report = registry.select(_identity_package())
    assert selected is backend
    assert device.device_id == "cpu"
    assert report.supported


@pytest.mark.parametrize("backend", [AscendBackend(), HorizonBackend()])
def test_vendor_stub_returns_structured_unsupported_report(backend: object) -> None:
    package = _identity_package()
    device = DeviceInfo(
        backend=backend.name,  # type: ignore[attr-defined]
        device_id="device:0",
        kind="accelerator",
        vendor="test",
        name="unavailable",
    )
    report = backend.supports(package, device)  # type: ignore[attr-defined]
    assert not report.supported
    assert report.reasons
