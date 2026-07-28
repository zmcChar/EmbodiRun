from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.backends import TorchBackendSession, TorchCudaBackend  # noqa: E402
from embodied_runtime.backends.torch_cuda.compiler import (  # noqa: E402
    TorchArtifactPayload,
    make_payload,
)
from embodied_runtime.contracts import (  # noqa: E402
    BackendExecutionError,
    CompileOptions,
    DeviceInfo,
    ExecutionContext,
    IterativeFlowPlan,
    ModelPackage,
    ModelSpec,
    RequestCancelledError,
    UnsupportedBackendError,
)


class TinyStages(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def stage(self, payload: dict[str, object]) -> dict[str, object]:
        nested = payload["nested"]
        assert isinstance(nested, Nested)
        integer = payload["integer"]
        assert isinstance(integer, torch.Tensor)
        return {
            "value": nested.value * self.scale,
            "integer": integer,
            "history": nested.history,
        }


class TrackingTinyStages(TinyStages):
    def __init__(self) -> None:
        super().__init__()
        self.moves: list[str] = []

    def to(self, *args: object, **kwargs: object) -> TrackingTinyStages:
        device = kwargs.get("device", args[0] if args else None)
        self.moves.append(str(device))
        super().to(*args, **kwargs)
        return self


class FailFirstMoveStages(TinyStages):
    def __init__(self) -> None:
        super().__init__()
        self.move_attempts = 0

    def to(self, *args: object, **kwargs: object) -> FailFirstMoveStages:
        self.move_attempts += 1
        if self.move_attempts == 1:
            raise RuntimeError("deliberate first placement failure")
        super().to(*args, **kwargs)
        return self


@dataclass(frozen=True)
class Nested:
    value: torch.Tensor
    history: tuple[torch.Tensor, ...]


def _package(module: TinyStages | None = None) -> ModelPackage:
    stage = module.stage if module is not None else (lambda payload: payload)
    return ModelPackage(
        spec=ModelSpec(model_id="tiny", family="test"),
        entrypoints={name: stage for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
        metadata={} if module is None else {"runtime_module": module},
    )


def _context() -> ExecutionContext:
    return ExecutionContext(request_ids=("request-1",), stage="test")


def _cpu_device(backend: TorchCudaBackend) -> DeviceInfo:
    return next(device for device in backend.probe() if device.device_id == "cpu")


def test_eager_session_moves_nested_tensors_and_runtime_module_dtype() -> None:
    backend = TorchCudaBackend()
    module = TinyStages()
    artifact = backend.compile(
        _package(module),
        _cpu_device(backend),
        CompileOptions(mode="eager", dtype="float64"),
    )
    # Compilation is a reusable recipe and must not mutate package-owned state.
    assert module.scale.dtype == torch.float32
    session = backend.load(artifact)
    output = session.submit(
        "encode_prefix",
        {
            "nested": Nested(
                torch.tensor([1.0], dtype=torch.float32),
                (torch.tensor([3.0], dtype=torch.float32),),
            ),
            "integer": torch.tensor([7], dtype=torch.int64),
        },
        _context(),
    )
    assert torch.equal(output["value"], torch.tensor([2.0], dtype=torch.float64))
    assert output["history"][0].dtype == torch.float64
    assert output["integer"].dtype == torch.int64
    assert module.scale.dtype == torch.float64
    assert session.package_id == artifact.package.package_id
    assert artifact.payload.package_id == artifact.package.package_id
    assert session.memory_stats().total_bytes is None or session.memory_stats().total_bytes > 0
    session.close()


def test_runtime_module_has_one_live_session_and_artifact_can_reload() -> None:
    backend = TorchCudaBackend()
    module = TinyStages()
    package = _package(module)
    device = _cpu_device(backend)
    first_artifact = backend.compile(package, device, CompileOptions(mode="eager"))
    second_artifact = backend.compile(package, device, CompileOptions(mode="eager"))
    artifact_entrypoints = dict(first_artifact.payload.entrypoints)
    artifact_eager_entrypoints = dict(first_artifact.payload.eager_entrypoints)

    first = backend.load(first_artifact)
    with pytest.raises(UnsupportedBackendError, match="already bound"):
        backend.load(first_artifact)
    with pytest.raises(UnsupportedBackendError, match="already bound"):
        backend.load(second_artifact)

    first.close()
    assert first_artifact.payload.runtime_module is module
    assert first_artifact.payload.entrypoints == artifact_entrypoints
    assert first_artifact.payload.eager_entrypoints == artifact_eager_entrypoints

    reloaded = backend.load(first_artifact)
    output = reloaded.submit(
        "encode_prefix",
        {
            "nested": Nested(torch.tensor([2.0]), (torch.tensor([3.0]),)),
            "integer": torch.tensor([1]),
        },
        _context(),
    )
    torch.testing.assert_close(output["value"], torch.tensor([4.0]))
    reloaded.close()


def test_runtime_module_dtype_policy_is_persistent_after_close() -> None:
    backend = TorchCudaBackend()
    module = TinyStages()
    package = _package(module)
    device = _cpu_device(backend)
    float64_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype="fp64"),
    )
    same_policy_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype="double"),
    )
    float32_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype="float32"),
    )
    preserve_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype=None),
    )

    first = backend.load(float64_artifact)
    first.close()
    same_policy = backend.load(same_policy_artifact)
    same_policy.close()

    with pytest.raises(UnsupportedBackendError, match="dtype policy.*torch.float64"):
        backend.load(float32_artifact)
    with pytest.raises(UnsupportedBackendError, match="dtype policy.*preserve"):
        backend.load(preserve_artifact)


def test_failed_first_load_does_not_lock_runtime_module_dtype_policy() -> None:
    backend = TorchCudaBackend()
    module = FailFirstMoveStages()
    package = _package(module)
    device = _cpu_device(backend)
    failed_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype="float64"),
    )
    preserve_artifact = backend.compile(
        package,
        device,
        CompileOptions(mode="eager", dtype=None),
    )

    with pytest.raises(RuntimeError, match="first placement failure"):
        backend.load(failed_artifact)

    session = backend.load(preserve_artifact)
    assert module.scale.dtype == torch.float32
    session.close()
    with pytest.raises(UnsupportedBackendError, match="dtype policy.*torch.float64"):
        backend.load(failed_artifact)


def test_session_observes_cancellation_and_close() -> None:
    backend = TorchCudaBackend()
    session = backend.load(backend.compile(_package(), _cpu_device(backend), CompileOptions()))
    context = _context()
    context.cancellation.set()
    with pytest.raises(RequestCancelledError):
        session.submit("encode_prefix", {"value": torch.ones(1)}, context)

    session.close()
    session.close()
    with pytest.raises(BackendExecutionError, match="closed"):
        session.submit("encode_prefix", {}, _context())


def test_unknown_entrypoint_is_a_backend_execution_error() -> None:
    backend = TorchCudaBackend()
    session = backend.load(backend.compile(_package(), _cpu_device(backend), CompileOptions()))
    with pytest.raises(BackendExecutionError, match="unknown model entrypoint"):
        session.submit("missing", {}, _context())


def test_real_torch_compile_mode_executes_portable_entrypoint() -> None:
    backend = TorchCudaBackend()
    session = backend.load(
        backend.compile(
            _package(),
            _cpu_device(backend),
            CompileOptions(
                mode="compile",
                options={
                    "compiler_backend": "eager",
                    "fallback_to_eager": False,
                },
            ),
        )
    )
    tensor = torch.tensor([1.0])
    output = session.submit("encode_prefix", {"value": tensor}, _context())
    torch.testing.assert_close(output["value"], tensor)
    assert session.actual_mode == "compile"
    assert not session.compile_failures


def test_deferred_torch_compile_failure_falls_back_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TorchCudaBackend()
    calls = {"compiled": 0, "eager": 0}

    def eager(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calls["eager"] += 1
        return {"value": payload["value"] + 1}

    package = ModelPackage(
        spec=ModelSpec(model_id="fallback", family="test"),
        entrypoints={name: eager for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )

    def fake_compile(function: object, **kwargs: object) -> object:
        del function, kwargs

        def fail(payload: object) -> object:
            del payload
            calls["compiled"] += 1
            raise RuntimeError("compiler specialization failed")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    artifact = backend.compile(
        package,
        _cpu_device(backend),
        CompileOptions(mode="compile", options={"fallback_to_eager": True}),
    )
    session = backend.load(artifact)
    first = session.submit("encode_prefix", {"value": torch.tensor(1)}, _context())
    second = session.submit("encode_prefix", {"value": torch.tensor(2)}, _context())
    assert first["value"].item() == 2
    assert second["value"].item() == 3
    assert calls == {"compiled": 1, "eager": 2}
    assert "encode_prefix" in session.compile_failures
    assert session.actual_mode == "mixed"
    # Runtime specialization failure belongs to this session, not the
    # reusable artifact or its compile-time metadata snapshot.
    assert artifact.payload.compile_failures == {}
    assert artifact.metadata["compile_failures"] == {}
    assert artifact.metadata["actual_mode"] == "compile"
    session.close()

    reloaded = backend.load(artifact)
    reloaded.submit("encode_prefix", {"value": torch.tensor(3)}, _context())
    assert calls == {"compiled": 2, "eager": 3}
    assert "encode_prefix" in reloaded.compile_failures
    reloaded.close()


def test_deferred_fallback_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TorchCudaBackend()
    calls = {"compiled": 0, "eager": 0}

    def eager(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        calls["eager"] += 1
        return payload

    package = ModelPackage(
        spec=ModelSpec(model_id="no-implicit-retry", family="test"),
        entrypoints={name: eager for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )

    def fake_compile(function: object, **kwargs: object) -> object:
        del function, kwargs

        def fail(payload: object) -> object:
            del payload
            calls["compiled"] += 1
            raise RuntimeError("do not execute twice")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    session = backend.load(
        backend.compile(package, _cpu_device(backend), CompileOptions(mode="compile"))
    )
    with pytest.raises(BackendExecutionError, match="do not execute twice"):
        session.submit("encode_prefix", {"value": torch.tensor(1)}, _context())
    assert calls == {"compiled": 1, "eager": 0}
    assert session.actual_mode == "compile"
    assert not session.compile_failures
    session.close()


def test_compile_failure_is_explicit_when_fallback_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TorchCudaBackend()

    def fake_compile(function: object, **kwargs: object) -> object:
        del function, kwargs

        def fail(payload: object) -> object:
            del payload
            raise RuntimeError("deliberate compile failure")

        return fail

    monkeypatch.setattr(torch, "compile", fake_compile)
    session = backend.load(
        backend.compile(
            _package(),
            _cpu_device(backend),
            CompileOptions(mode="compile", options={"fallback_to_eager": False}),
        )
    )
    with pytest.raises(BackendExecutionError, match="deliberate compile failure"):
        session.submit("encode_prefix", {}, _context())
    session.close()


def test_add_scaled_executes_recursive_tensor_arithmetic_in_session() -> None:
    backend = TorchCudaBackend()
    session = backend.load(backend.compile(_package(), _cpu_device(backend), CompileOptions()))
    output = session.add_scaled(
        {
            "tensor": torch.tensor([1.0, 2.0]),
            "nested": [torch.tensor([3.0])],
            "pair": (torch.tensor([4.0]),),
        },
        {
            "tensor": torch.tensor([2.0, -2.0]),
            "nested": [torch.tensor([4.0])],
            "pair": (torch.tensor([-2.0]),),
        },
        0.5,
        _context(),
    )
    torch.testing.assert_close(output["tensor"], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(output["nested"][0], torch.tensor([5.0]))
    torch.testing.assert_close(output["pair"][0], torch.tensor([3.0]))
    session.close()


def test_add_scaled_preserves_explicit_multiply_then_add_rounding() -> None:
    backend = TorchCudaBackend()
    session = backend.load(backend.compile(_package(), _cpu_device(backend), CompileOptions()))
    state = torch.tensor([30.813392639160156], dtype=torch.float32)
    update = torch.tensor([-137.39041137695312], dtype=torch.float32)
    scale = -0.27811792492866516
    expected = state + update * scale

    # This witness guards against replacing the declared operation with an
    # alpha-add/fused multiply-add that rounds differently.
    fused = torch.add(state, update, alpha=scale)
    assert not torch.equal(expected, fused)
    output = session.add_scaled(state, update, scale, _context())
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    session.close()


def _cuda_payload(*, synchronize_on_submit: bool = True) -> TorchArtifactPayload:
    entrypoints = {
        name: (lambda payload: payload) for name in IterativeFlowPlan().required_entrypoints()
    }
    return TorchArtifactPayload(
        package_id="cuda-sync-test",
        entrypoints=dict(entrypoints),
        eager_entrypoints=dict(entrypoints),
        device_id="cuda:0",
        dtype=None,
        requested_mode="eager",
        actual_mode="eager",
        synchronize_on_submit=synchronize_on_submit,
    )


def test_cuda_submit_and_add_scaled_synchronize_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    device = DeviceInfo(
        backend="torch_cuda",
        device_id="cuda:0",
        kind="accelerator",
        vendor="NVIDIA",
        name="fake CUDA",
    )
    session = TorchBackendSession(torch, device, _cuda_payload())
    session.submit("encode_prefix", {}, _context())
    session.add_scaled(1.0, 2.0, 0.5, _context())
    assert calls == ["cuda:0", "cuda:0"]
    session.close()

    unsynchronized = TorchBackendSession(
        torch,
        device,
        _cuda_payload(synchronize_on_submit=False),
    )
    unsynchronized.submit("encode_prefix", {}, _context())
    unsynchronized.add_scaled(1.0, 2.0, 0.5, _context())
    assert calls == ["cuda:0", "cuda:0"]
    unsynchronized.close()


def test_cuda_payload_defaults_to_offload_and_cache_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = TrackingTinyStages()
    package = _package(module)
    payload = make_payload(
        torch,
        package,
        CompileOptions(mode="eager"),
        device_id="cuda:0",
    )
    assert payload.synchronize_on_submit
    assert payload.offload_module_on_close
    assert payload.empty_cache_on_close
    assert module.moves == []

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(f"sync:{device}"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    device = DeviceInfo(
        backend="torch_cuda",
        device_id="cuda:0",
        kind="accelerator",
        vendor="NVIDIA",
        name="fake CUDA",
    )
    session = TorchBackendSession(torch, device, payload)
    session.close()
    assert calls == ["sync:cuda:0", "empty"]
    assert module.moves == ["cpu"]


def test_cuda_graph_configuration_is_explicit_and_backend_local() -> None:
    backend = TorchCudaBackend()
    package = _package()
    cpu = _cpu_device(backend)

    with pytest.raises(UnsupportedBackendError, match="require a CUDA device"):
        backend.compile(
            package,
            cpu,
            CompileOptions(
                mode="eager",
                options={"cuda_graph_entrypoints": ("encode_prefix",)},
            ),
        )
    with pytest.raises(UnsupportedBackendError, match="unknown CUDA Graph entrypoint"):
        make_payload(
            torch,
            package,
            CompileOptions(
                mode="eager",
                options={"cuda_graph_entrypoints": ("missing",)},
            ),
            device_id="cuda:0",
        )
    with pytest.raises(UnsupportedBackendError, match="require PyTorch eager mode"):
        make_payload(
            torch,
            package,
            CompileOptions(
                mode="compile",
                options={"cuda_graph_entrypoints": ("encode_prefix",)},
            ),
            device_id="cuda:0",
        )
    with pytest.raises(UnsupportedBackendError, match="warmup_steps"):
        make_payload(
            torch,
            package,
            CompileOptions(
                mode="eager",
                options={
                    "cuda_graph_entrypoints": ("encode_prefix",),
                    "cuda_graph_warmup_steps": 0,
                },
            ),
            device_id="cuda:0",
        )

    payload = make_payload(torch, package, CompileOptions(), device_id="cpu")
    assert not payload.cuda_graph.enabled


def test_cuda_graph_dynamic_scalar_configuration_rejects_unknown_entrypoint() -> None:
    with pytest.raises(
        UnsupportedBackendError,
        match="require selected CUDA Graph entrypoint.*missing",
    ):
        make_payload(
            torch,
            _package(),
            CompileOptions(
                mode="eager",
                options={
                    "cuda_graph_entrypoints": ("encode_prefix",),
                    "cuda_graph_dynamic_scalar_inputs": {
                        "missing": ("time",),
                    },
                },
            ),
            device_id="cuda:0",
        )


def test_cuda_graph_dynamic_scalar_reuses_graph_and_updates_value_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    def stage(payload: dict[str, object]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        value = payload["value"]
        time = payload["time"]
        assert isinstance(value, torch.Tensor)
        return {"value": value * time}

    package = ModelPackage(
        spec=ModelSpec(model_id="dynamic-scalar-graph-test", family="test"),
        entrypoints={name: stage for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )
    backend = TorchCudaBackend()
    device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            device,
            CompileOptions(
                mode="eager",
                options={
                    "cuda_graph_entrypoints": ("encode_prefix",),
                    "cuda_graph_dynamic_scalar_inputs": {
                        "encode_prefix": ("time",),
                    },
                },
            ),
        )
    )

    first = session.submit(
        "encode_prefix",
        {"value": torch.tensor([3.0]), "time": 1.0},
        _context(),
    )
    second = session.submit(
        "encode_prefix",
        {"value": torch.tensor([3.0]), "time": 2.0},
        _context(),
    )

    torch.testing.assert_close(first["value"].cpu(), torch.tensor([3.0]))
    torch.testing.assert_close(second["value"].cpu(), torch.tensor([6.0]))
    assert calls == 2  # one warmup plus one capture-time Python invocation
    assert session.cuda_graph_stats == {
        "captures": 1,
        "replays": 1,
        "fallbacks": 0,
        "cached_graphs": 1,
    }
    assert not session.cuda_graph_failures
    session.close()


def test_cuda_graph_capture_replay_and_output_ownership_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    def stage(payload: dict[str, object]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        value = payload["value"]
        bias = payload["bias"]
        assert isinstance(value, torch.Tensor)
        assert isinstance(bias, float)
        return {"value": value * 2 + bias}

    package = ModelPackage(
        spec=ModelSpec(model_id="generic-graph-test", family="test"),
        entrypoints={name: stage for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )
    backend = TorchCudaBackend()
    device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    artifact = backend.compile(
        package,
        device,
        CompileOptions(
            mode="eager",
            options={
                "cuda_graph_entrypoints": ("encode_prefix",),
                "cuda_graph_max_graphs": 3,
                "empty_cache_on_close": True,
            },
        ),
    )
    assert artifact.metadata["cuda_graph_entrypoints"] == ("encode_prefix",)
    session = backend.load(artifact)

    first = session.submit(
        "encode_prefix",
        {"value": torch.tensor([1.0]), "bias": 1.0},
        _context(),
    )
    first_snapshot = first["value"].clone()
    assert calls == 2  # one warmup plus one capture-time Python invocation
    torch.testing.assert_close(first["value"].cpu(), torch.tensor([3.0]))

    second = session.submit(
        "encode_prefix",
        {"value": torch.tensor([4.0]), "bias": 1.0},
        _context(),
    )
    assert calls == 2
    torch.testing.assert_close(second["value"].cpu(), torch.tensor([9.0]))
    # The returned result must not alias graph-owned output buffers.
    torch.testing.assert_close(first["value"], first_snapshot)

    noncontiguous = torch.tensor(
        [[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]],
        device="cuda:0",
    ).transpose(0, 1)
    assert not noncontiguous.is_contiguous()
    wider = session.submit(
        "encode_prefix",
        {"value": noncontiguous, "bias": 1.0},
        _context(),
    )
    assert calls == 4
    torch.testing.assert_close(
        wider["value"].cpu(),
        torch.tensor([[5.0, 7.0], [9.0, 11.0], [13.0, 15.0]]),
    )

    again = session.submit(
        "encode_prefix",
        {"value": torch.tensor([5.0]), "bias": 1.0},
        _context(),
    )
    assert calls == 4
    torch.testing.assert_close(again["value"].cpu(), torch.tensor([11.0]))

    # Python leaves are capture-time constants and therefore part of the
    # signature; changing one must never replay a graph with a frozen value.
    changed_constant = session.submit(
        "encode_prefix",
        {"value": torch.tensor([5.0]), "bias": 2.0},
        _context(),
    )
    assert calls == 6
    torch.testing.assert_close(changed_constant["value"].cpu(), torch.tensor([12.0]))
    changed_constant_again = session.submit(
        "encode_prefix",
        {"value": torch.tensor([1.0]), "bias": 2.0},
        _context(),
    )
    assert calls == 6
    torch.testing.assert_close(changed_constant_again["value"].cpu(), torch.tensor([4.0]))
    assert session.cuda_graph_stats == {
        "captures": 3,
        "replays": 3,
        "fallbacks": 0,
        "cached_graphs": 3,
    }
    assert not session.cuda_graph_failures

    # Only the explicit allowlist is captured.
    session.submit(
        "finalize",
        {"value": torch.tensor([1.0]), "bias": 1.0},
        _context(),
    )
    assert calls == 7
    session.close()
    assert session.cuda_graph_stats["cached_graphs"] == 0

    reloaded = backend.load(artifact)
    assert reloaded.cuda_graph_stats == {
        "captures": 0,
        "replays": 0,
        "fallbacks": 0,
        "cached_graphs": 0,
    }
    reloaded.close()


def test_cuda_graph_capture_failure_falls_back_once_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    def capture_incompatible(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        output = payload["value"] + 1
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("deliberately not graph-safe")
        return {"value": output}

    package = ModelPackage(
        spec=ModelSpec(model_id="generic-fallback-test", family="test"),
        entrypoints={
            name: capture_incompatible for name in IterativeFlowPlan().required_entrypoints()
        },
        plan=IterativeFlowPlan(),
    )
    backend = TorchCudaBackend()
    device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            device,
            CompileOptions(
                mode="eager",
                options={"cuda_graph_entrypoints": ("encode_prefix",)},
            ),
        )
    )

    first = session.submit("encode_prefix", {"value": torch.tensor([1.0])}, _context())
    second = session.submit("encode_prefix", {"value": torch.tensor([3.0])}, _context())
    torch.testing.assert_close(first["value"].cpu(), torch.tensor([2.0]))
    torch.testing.assert_close(second["value"].cpu(), torch.tensor([4.0]))
    assert calls == 4
    assert session.cuda_graph_stats["captures"] == 0
    assert session.cuda_graph_stats["fallbacks"] == 2
    assert "deliberately not graph-safe" in session.cuda_graph_failures["encode_prefix"]
    session.close()


def test_cuda_graph_shared_storage_conservatively_falls_back_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    def stage(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"value": payload["left"] + payload["right"]}

    package = ModelPackage(
        spec=ModelSpec(model_id="alias-fallback-test", family="test"),
        entrypoints={name: stage for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )
    backend = TorchCudaBackend()
    device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            device,
            CompileOptions(
                options={"cuda_graph_entrypoints": ("encode_prefix",)},
            ),
        )
    )
    shared = torch.tensor([1.0, 2.0], device="cuda:0")
    output = session.submit(
        "encode_prefix",
        {"left": shared, "right": shared},
        _context(),
    )
    torch.testing.assert_close(output["value"].cpu(), torch.tensor([2.0, 4.0]))
    assert session.cuda_graph_stats["captures"] == 0
    assert session.cuda_graph_stats["fallbacks"] == 1
    assert "share storage" in session.cuda_graph_failures["encode_prefix"]
    session.close()


def test_cuda_graph_warmup_model_error_is_not_retried_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    calls = 0

    def fail(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"ordinary model failure for {payload['value'].shape}")

    package = ModelPackage(
        spec=ModelSpec(model_id="warmup-error-test", family="test"),
        entrypoints={name: fail for name in IterativeFlowPlan().required_entrypoints()},
        plan=IterativeFlowPlan(),
    )
    backend = TorchCudaBackend()
    device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            package,
            device,
            CompileOptions(
                options={"cuda_graph_entrypoints": ("encode_prefix",)},
            ),
        )
    )
    with pytest.raises(BackendExecutionError, match="ordinary model failure"):
        session.submit("encode_prefix", {"value": torch.ones(1)}, _context())
    assert calls == 1
    assert session.cuda_graph_stats["fallbacks"] == 0
    assert not session.cuda_graph_failures
    session.close()


def test_cuda_eager_smoke_if_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    backend = TorchCudaBackend()
    cuda_device = next(device for device in backend.probe() if device.device_id == "cuda:0")
    session = backend.load(
        backend.compile(
            _package(),
            cuda_device,
            CompileOptions(mode="eager", dtype="float32"),
        )
    )
    output = session.submit(
        "encode_prefix",
        {"value": torch.ones(2)},
        _context(),
    )
    assert output["value"].device.type == "cuda"
    assert session.memory_stats().total_bytes is not None
    session.close()
