"""PyTorch compiler tests that do not require PyTorch to be installed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodied_runtime.backends import CompileOptions, UnsupportedBackendError
from embodied_runtime.backends.torch_cuda.compiler import (
    compile_entrypoints,
    make_payload,
    normalize_mode,
    resolve_dtype,
)
from embodied_runtime.backends.torch_cuda.compiler_entrypoints import (
    compile_entrypoints as focused_compile_entrypoints,
)
from embodied_runtime.backends.torch_cuda.compiler_validation import (
    normalize_mode as focused_normalize_mode,
)


class _FakeTorch:
    float32 = object()
    float16 = object()
    bfloat16 = object()
    float64 = object()
    nn = SimpleNamespace(Module=type("Module", (), {}))


def test_established_compiler_imports_use_focused_implementations() -> None:
    assert compile_entrypoints is focused_compile_entrypoints
    assert normalize_mode is focused_normalize_mode


def test_compiler_validation_and_eager_payload_need_no_real_torch() -> None:
    torch = _FakeTorch()
    entrypoint = lambda value: value
    package = SimpleNamespace(
        package_id="fake-package",
        entrypoints={"forward": entrypoint},
        metadata={},
    )

    payload = make_payload(torch, package, CompileOptions(dtype="fp16"), device_id="cpu")

    assert normalize_mode(" torch_eager ") == "eager"
    assert normalize_mode("torch.compile") == "compile"
    assert resolve_dtype(torch, "torch.float_16") is torch.float16
    assert payload.package_id == "fake-package"
    assert payload.entrypoints == {"forward": entrypoint}
    assert payload.actual_mode == "eager"
    assert not payload.cuda_graph.enabled
    with pytest.raises(UnsupportedBackendError, match="unsupported PyTorch execution mode"):
        normalize_mode("invalid")


def test_compiler_selects_mixed_fallback_without_real_torch() -> None:
    good = lambda value: value
    bad = lambda value: value

    class FakeCompilerTorch:
        @staticmethod
        def compile(function, **kwargs):
            assert kwargs == {"fullgraph": False, "dynamic": False}
            if function is bad:
                raise RuntimeError("cannot compile")
            return lambda value: function(value)

    package = SimpleNamespace(entrypoints={"good": good, "bad": bad})
    options = CompileOptions(mode="compile", options={"fallback_to_eager": True})

    selected, eager, failures, actual_mode = compile_entrypoints(
        FakeCompilerTorch(), package, options
    )

    assert selected["bad"] is bad
    assert selected["good"] is not good
    assert eager == {"good": good, "bad": bad}
    assert failures == {"bad": "RuntimeError: cannot compile"}
    assert actual_mode == "mixed"
