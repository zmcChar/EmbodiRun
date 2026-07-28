from __future__ import annotations

import pytest

from embodied_runtime.contracts import ModelAdapter, ModelPackageError
from embodied_runtime.models import available_models, get_model_adapter, register_model


def test_builtin_models_are_listed_without_loading_weights() -> None:
    assert {"pi05", "toy_flow", "toy_single_forward"}.issubset(available_models())


def test_registry_lazily_constructs_toy_adapter() -> None:
    pytest.importorskip("torch")
    adapter = get_model_adapter("toy_flow")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.describe().family == "toy_flow"


def test_unknown_adapter_lists_available_models() -> None:
    with pytest.raises(ModelPackageError, match="toy_flow"):
        get_model_adapter("definitely_missing")


def test_registry_constructs_single_forward_adapter() -> None:
    pytest.importorskip("torch")
    adapter = get_model_adapter("toy_single_forward")
    assert isinstance(adapter, ModelAdapter)
    assert adapter.describe().family == "toy_single_forward"


def test_duplicate_registration_is_rejected() -> None:
    class Adapter:
        pass

    register_model("_model_test_duplicate", Adapter)
    with pytest.raises(ValueError, match="already registered"):
        register_model("_model_test_duplicate", Adapter)
