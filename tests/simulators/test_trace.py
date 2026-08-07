from __future__ import annotations

import json

import pytest

from embodied_runtime.robots.action import RobotAction
from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.simulators import EpisodeStep, TraceRecorder


def test_trace_records_ordered_episode_and_returns_json_compatible_summary() -> None:
    reset_values = {"joints": [0.0, 0.0]}
    reset_observation = RobotObservation(
        timestamp_s=0.0,
        values=reset_values,
        metadata={"camera": "front"},
    )
    action = RobotAction(timestamp_s=0.1, values={"joints": [0.2, 0.3]})
    outcome = EpisodeStep(
        observation=RobotObservation(timestamp_s=0.2, values={"joints": [0.2, 0.3]}),
        reward=1,
        terminated=True,
        truncated=False,
        success=True,
        subgoal_progress=1,
        info={"metrics": {"distance": 0.0}},
    )
    disturbance_details = {"offset": [0.1, 0.0, 0.0]}
    recorder = TraceRecorder(task=" composite-pick ", seed=17)

    recorder.record_reset(reset_observation)
    recorder.record_disturbance("move-target", disturbance_details)
    recorder.record_step(action, outcome)
    reset_values["joints"][0] = 99.0
    disturbance_details["offset"][0] = 99.0

    trace = recorder.snapshot()
    summary = trace.summary()

    assert trace.task == "composite-pick"
    assert [event.kind for event in trace.events] == ["reset", "disturbance", "step"]
    assert [event.sequence for event in trace.events] == [0, 1, 2]
    assert len(trace.steps) == 1
    assert len(trace.disturbances) == 1
    assert summary["total_reward"] == 1.0
    assert summary["success"] is True
    assert summary["events"][0]["observation"]["values"]["joints"] == [0.0, 0.0]
    assert summary["events"][1]["details"]["offset"] == [0.1, 0.0, 0.0]
    assert json.loads(json.dumps(summary, allow_nan=False, sort_keys=True)) == json.loads(
        json.dumps(recorder.summary(), allow_nan=False, sort_keys=True)
    )


def test_trace_recorder_enforces_episode_lifecycle() -> None:
    recorder = TraceRecorder()
    observation = RobotObservation(timestamp_s=0.0, values={})
    action = RobotAction(timestamp_s=0.1, values={})
    terminal = EpisodeStep(observation, 0.0, False, True)

    with pytest.raises(RuntimeError, match="reset"):
        recorder.record_disturbance("move-target")
    with pytest.raises(RuntimeError, match="reset"):
        recorder.snapshot()

    recorder.record_reset(observation)
    with pytest.raises(RuntimeError, match="already"):
        recorder.record_reset(observation)
    recorder.record_step(action, terminal)
    with pytest.raises(RuntimeError, match="terminal"):
        recorder.record_disturbance("move-target")
    with pytest.raises(RuntimeError, match="terminal"):
        recorder.record_step(action, terminal)


def test_trace_summary_normalizes_non_json_tensor_like_values() -> None:
    class TensorLike:
        def tolist(self) -> list[float]:
            return [1.0, 2.0]

    recorder = TraceRecorder()
    recorder.record_reset(
        RobotObservation(
            timestamp_s=float("inf"),
            values={"tensor": TensorLike(), "labels": frozenset({"b", "a"})},
        )
    )

    summary = recorder.summary()

    assert summary["events"][0]["observation"]["timestamp_s"] == "Infinity"
    assert summary["events"][0]["observation"]["values"] == {
        "labels": ["a", "b"],
        "tensor": [1.0, 2.0],
    }
    json.dumps(summary, allow_nan=False)
