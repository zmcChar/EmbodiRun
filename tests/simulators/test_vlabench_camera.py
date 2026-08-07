from __future__ import annotations

from typing import Any

import pytest

from embodied_runtime.simulators import (
    VLABenchCameraMappingError,
    VLABenchSemanticCameraMixin,
    compiled_camera_names,
    resolve_vlabench_camera_indices,
)

np = pytest.importorskip("numpy")


class _FakeModel:
    def __init__(self, names: list[str | None]) -> None:
        self.names = names
        self.ncam = len(names)

    def id2name(self, index: int, kind: str) -> str | None:
        assert kind == "camera"
        return self.names[index]


class _FakeRobot:
    def get_ee_state(self, physics: Any) -> Any:
        assert isinstance(physics.model, _FakeModel)
        return np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0])


class _FakeInnerEnvironment:
    def __init__(self, names: list[str | None]) -> None:
        self.physics = type("_Physics", (), {"model": _FakeModel(names)})()
        self.robot = _FakeRobot()
        self.rendered_camera_ids: list[int] = []

    def render(self, *, camera_id: int, height: int, width: int) -> Any:
        self.rendered_camera_ids.append(camera_id)
        return np.full((height, width, 3), camera_id, dtype=np.uint8)


class _ObservationHarness(VLABenchSemanticCameraMixin):
    def __init__(self, names: list[str | None]) -> None:
        self._env = _FakeInnerEnvironment(names)
        self._robot_base_xyz = np.array([0.5, 1.0, 1.5])
        self.obs_type = "pixels_agent_pos"
        self.render_resolution = (2, 3)


def test_resolves_dataset_views_from_compiled_names_not_position() -> None:
    names = ["right", "left", "forward", "robot/Franka_wrist_cam"]

    indices = resolve_vlabench_camera_indices(names)

    assert indices == {
        "image": 1,
        "second_image": 0,
        "wrist_image": 3,
    }
    assert indices["wrist_image"] != names.index("forward")


def test_semantic_observation_renders_left_right_and_true_wrist() -> None:
    environment = _ObservationHarness(["right", "left", "forward", "robot/Franka_wrist_cam"])

    observation = environment._get_obs()

    assert environment._env.rendered_camera_ids == [1, 0, 3]
    assert np.all(observation["pixels"]["image"] == 1)
    assert np.all(observation["pixels"]["second_image"] == 0)
    assert np.all(observation["pixels"]["wrist_image"] == 3)
    np.testing.assert_allclose(
        observation["agent_pos"],
        [0.5, 1.0, 1.5, 0.0, 0.0, 0.0, 1.0],
    )


def test_reads_names_in_compiled_render_order() -> None:
    model = _FakeModel(["right", "left", "forward", "Franka_wrist_cam"])

    assert compiled_camera_names(model) == model.names


@pytest.mark.parametrize(
    "names, missing_key",
    [
        (["right", "left", "forward"], "wrist_image"),
        (["right", "left", "left", "Franka_wrist_cam"], "image"),
    ],
)
def test_rejects_missing_or_ambiguous_semantic_cameras(
    names: list[str],
    missing_key: str,
) -> None:
    with pytest.raises(VLABenchCameraMappingError, match=missing_key):
        resolve_vlabench_camera_indices(names)
