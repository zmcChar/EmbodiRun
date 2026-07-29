from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodied_runtime.backends.torch_cuda import TorchCudaBackend
from embodied_runtime.contracts import (
    CompileOptions,
    ModelAdapter,
    ModelPackageError,
    RawRequest,
    SingleForwardPlan,
)
from embodied_runtime.engine import ExecutionEngine
from embodied_runtime.models import available_models, get_model_adapter
from embodied_runtime.models.vla.gr00t_n17 import (
    DEFAULT_EMBODIMENT_TAG,
    DEFAULT_LANGUAGE_KEY,
    Gr00tN17Adapter,
    load_gr00t_n17,
    synthetic_droid_request,
)
from embodied_runtime.models.vla.gr00t_n17 import adapter as adapter_module


class _FakeRuntimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(max_action_dim=132)


class _FakeGr00tPolicy:
    def __init__(self) -> None:
        self.model = _FakeRuntimeModel()
        self.embodiment_tag = SimpleNamespace(value="oxe_droid_relative_eef_relative_joint")
        self.language_key = DEFAULT_LANGUAGE_KEY
        self.modality_configs = {
            "video": SimpleNamespace(
                modality_keys=("exterior_image_1_left", "wrist_image_left"),
                delta_indices=(-15, 0),
            ),
            "state": SimpleNamespace(
                modality_keys=("eef_9d", "gripper_position", "joint_position"),
                delta_indices=(0,),
            ),
            "action": SimpleNamespace(
                modality_keys=("eef_9d", "gripper_position"),
                delta_indices=tuple(range(4)),
            ),
            "language": SimpleNamespace(
                modality_keys=(DEFAULT_LANGUAGE_KEY,),
                delta_indices=(0,),
            ),
        }
        self.observations: list[dict] = []

    def get_action(self, observation):
        self.observations.append(observation)
        batch_size = observation["video"]["exterior_image_1_left"].shape[0]
        seed = observation["state"]["gripper_position"][:, 0, 0]
        eef = np.stack(
            [np.stack((seed + step, seed + step + 0.5), axis=-1) for step in range(4)],
            axis=1,
        ).astype(np.float32)
        gripper = np.repeat(seed[:, None, None], 4, axis=1).astype(np.float32)
        assert eef.shape == (batch_size, 4, 2)
        return {"eef_9d": eef, "gripper_position": gripper}, {"source": "fake"}


def _request(seed: float, *, prompt: str) -> RawRequest:
    video = {
        "exterior_image_1_left": np.full((2, 4, 5, 3), int(seed), dtype=np.uint8),
        "wrist_image_left": np.full((1, 2, 4, 5, 3), int(seed) + 1, dtype=np.uint8),
    }
    state = {
        "eef_9d": np.full((1, 9), seed, dtype=np.float32),
        "gripper_position": np.full((1, 1, 1), seed, dtype=np.float32),
        "joint_position": np.full((1, 7), seed, dtype=np.float32),
    }
    return RawRequest(
        observation={"video": video, "state": state},
        prompt=prompt,
    )


def _package():
    adapter = Gr00tN17Adapter()
    policy = _FakeGr00tPolicy()
    package = adapter.build_package(
        "injected-gr00t",
        policy=policy,
        action_dim=3,
        revision="test-revision",
    )
    return adapter, policy, package


def _cpu_device(backend: TorchCudaBackend):
    return next(device for device in backend.probe() if device.device_id == "cpu")


def test_registry_is_lazy_and_describes_default_droid_contract() -> None:
    assert "gr00t_n17" in available_models()

    adapter = get_model_adapter("gr00t_n17")

    assert isinstance(adapter, ModelAdapter)
    assert isinstance(adapter, Gr00tN17Adapter)
    spec = adapter.describe()
    assert spec.family == "gr00t_n17"
    assert spec.action_dim == 17
    assert spec.action_horizon == 40
    assert spec.metadata["embodiment_tag"] == DEFAULT_EMBODIMENT_TAG
    assert spec.metadata["language_key"] == DEFAULT_LANGUAGE_KEY


def test_initial_adapter_rejects_unimplemented_embodiment_contracts() -> None:
    with pytest.raises(ValueError, match="supports only"):
        Gr00tN17Adapter(embodiment_tag="LIBERO_PANDA")


def test_injected_policy_builds_formal_single_forward_package() -> None:
    adapter, policy, package = _package()

    assert isinstance(package.plan, SingleForwardPlan)
    assert package.metadata["runtime_module"] is policy.model
    assert package.entrypoint_specs["forward"].batchable
    assert package.spec.action_dim == 3
    assert package.spec.action_horizon == 4
    assert package.spec.revision == "test-revision"
    assert adapter.loaded_policy is policy


def test_preprocess_collate_unbatch_and_mapping_action_chunk() -> None:
    adapter, policy, package = _package()
    first = adapter.preprocess_one(_request(2.0, prompt="pick up the cup"))
    second_request = _request(7.0, prompt="move to the drawer")
    second_request.observation["language"] = {DEFAULT_LANGUAGE_KEY: [["explicit instruction"]]}
    second = adapter.preprocess_one(second_request)

    assert first["video"]["exterior_image_1_left"].shape == (1, 2, 4, 5, 3)
    assert first["state"]["eef_9d"].shape == (1, 1, 9)
    assert first["language"][DEFAULT_LANGUAGE_KEY] == [["pick up the cup"]]
    assert second["language"][DEFAULT_LANGUAGE_KEY] == [["explicit instruction"]]

    payload = adapter.collate((first, second))
    assert payload["video"]["wrist_image_left"].shape == (2, 2, 4, 5, 3)
    assert payload["state"]["joint_position"].shape == (2, 1, 7)
    assert payload["language"][DEFAULT_LANGUAGE_KEY] == [
        ["pick up the cup"],
        ["explicit instruction"],
    ]

    output = package.entrypoints["forward"](payload)
    samples = adapter.unbatch(output, batch_size=2)
    chunks = tuple(adapter.postprocess_one(sample) for sample in samples)

    assert len(policy.observations) == 1
    assert len(chunks) == 2
    assert set(chunks[0].actions) == {"eef_9d", "gripper_position"}
    assert chunks[0].actions["eef_9d"].shape == (4, 2)
    assert chunks[0].actions["gripper_position"].shape == (4, 1)
    np.testing.assert_allclose(chunks[0].actions["eef_9d"][0], [2.0, 2.5])
    np.testing.assert_allclose(chunks[1].actions["eef_9d"][3], [10.0, 10.5])
    assert chunks[0].metadata["policy_info"] == {"source": "fake"}


def test_synthetic_droid_request_matches_default_contract() -> None:
    request = synthetic_droid_request(
        "put the cup in the sink",
        image_height=12,
        image_width=16,
    )
    adapter = Gr00tN17Adapter()

    payload = adapter.preprocess_one(request)

    assert payload["video"]["exterior_image_1_left"].shape == (1, 2, 12, 16, 3)
    assert payload["video"]["wrist_image_left"].dtype == np.uint8
    assert payload["state"]["eef_9d"].shape == (1, 1, 9)
    np.testing.assert_array_equal(
        payload["state"]["eef_9d"][0, 0, 3:],
        np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32),
    )
    assert payload["state"]["gripper_position"].shape == (1, 1, 1)
    assert payload["state"]["joint_position"].shape == (1, 1, 7)
    assert payload["language"][DEFAULT_LANGUAGE_KEY] == [["put the cup in the sink"]]


def test_injected_policy_runs_through_cpu_backend_and_engine() -> None:
    adapter, policy, package = _package()
    payload = adapter.preprocess_one(_request(3.0, prompt="close the drawer"))

    backend = TorchCudaBackend()
    session = backend.load(
        backend.compile(
            package,
            _cpu_device(backend),
            CompileOptions(mode="eager", dtype=None),
        )
    )
    engine = ExecutionEngine(
        package,
        session,
        batcher=adapter.collate,
        splitter=adapter.unbatch,
    )
    try:
        result = engine.infer(payload)
        chunk = adapter.postprocess_one(result.output)
    finally:
        engine.close()

    assert policy.model.anchor.device.type == "cpu"
    assert result.metadata["model_id"] == "injected-gr00t"
    assert set(chunk.actions) == {"eef_9d", "gripper_position"}
    np.testing.assert_allclose(chunk.actions["eef_9d"][2], [5.0, 5.5])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda request: request.observation["video"].__setitem__(
                "wrist_image_left",
                np.zeros((2, 4, 5, 3), dtype=np.float32),
            ),
            "dtype uint8",
        ),
        (
            lambda request: request.observation["state"].__setitem__(
                "eef_9d",
                np.zeros((2, 1, 9), dtype=np.float32),
            ),
            "shape",
        ),
    ],
)
def test_preprocess_rejects_noncanonical_observations(mutate, message: str) -> None:
    adapter = Gr00tN17Adapter()
    request = _request(1.0, prompt="do the task")
    mutate(request)

    with pytest.raises(ModelPackageError, match=message):
        adapter.preprocess_one(request)


def test_collate_rejects_different_robot_keys() -> None:
    adapter = Gr00tN17Adapter()
    first = adapter.preprocess_one(_request(1.0, prompt="first"))
    second = adapter.preprocess_one(_request(2.0, prompt="second"))
    second["state"].pop("joint_position")

    with pytest.raises(ModelPackageError, match="keys differ"):
        adapter.collate((first, second))


def test_loader_accepts_local_directory_without_network(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeOfficialPolicy:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(adapter_module, "_require_gr00t_policy", lambda: FakeOfficialPolicy)
    policy = load_gr00t_n17(tmp_path, strict=False)

    assert isinstance(policy, FakeOfficialPolicy)
    assert calls == [
        {
            "embodiment_tag": DEFAULT_EMBODIMENT_TAG,
            "model_path": str(tmp_path.resolve()),
            "device": "cpu",
            "strict": False,
        }
    ]


def test_loader_resolves_hugging_face_repository_without_real_network(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    class FakeOfficialPolicy:
        def __init__(self, **kwargs) -> None:
            calls.append(("policy", kwargs))

    def fake_download(checkpoint, **kwargs):
        calls.append(("download", checkpoint, kwargs))
        return tmp_path

    monkeypatch.setattr(adapter_module, "_require_gr00t_policy", lambda: FakeOfficialPolicy)
    monkeypatch.setattr(adapter_module, "_download_hf_snapshot", fake_download)

    load_gr00t_n17(
        "nvidia/GR00T-N1.7-3B",
        cache_dir="/cache",
        revision="revision-a",
        local_files_only=True,
    )

    assert calls[0] == (
        "download",
        "nvidia/GR00T-N1.7-3B",
        {
            "cache_dir": "/cache",
            "revision": "revision-a",
            "local_files_only": True,
        },
    )
    assert calls[1][0] == "policy"
    assert calls[1][1]["model_path"] == str(tmp_path)


def test_loader_rejects_direct_accelerator_placement(tmp_path) -> None:
    with pytest.raises(ModelPackageError, match="backend owns device placement"):
        load_gr00t_n17(tmp_path, load_device="cuda:0")
