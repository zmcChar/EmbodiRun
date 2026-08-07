from __future__ import annotations

import pytest

from embodied_runtime.engine import (
    InferenceProvider,
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)
from embodied_runtime.engine.providers import ProviderRegistry
from embodied_runtime.models.spec import ModelSpec


class _SimpleProvider:
    def __init__(self) -> None:
        self._capabilities = ProviderCapabilities(
            name="fixture",
            runtime="fixture_runtime",
            model=ModelSpec(model_id="fixture", family="test"),
            is_remote=False,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def infer_async(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(request_id=request.request_id, output=request.payload)

    async def aclose(self) -> None:
        pass


def test_provider_registry_is_lazy_and_validates_created_provider() -> None:
    calls = 0

    def factory() -> InferenceProvider:
        nonlocal calls
        calls += 1
        return _SimpleProvider()

    registry = ProviderRegistry()
    registry.register(" Fixture ", factory)
    assert registry.names() == ("fixture",)
    assert calls == 0

    provider = registry.create("FIXTURE")
    assert isinstance(provider, InferenceProvider)
    assert calls == 1

    with pytest.raises(ValueError, match="already registered"):
        registry.register("fixture", factory)
    with pytest.raises(KeyError, match="registered providers: fixture"):
        registry.create("missing")

    registry.register("invalid", lambda: object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="does not satisfy InferenceProvider"):
        registry.create("invalid")
