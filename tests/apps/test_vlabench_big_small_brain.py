from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from embodied_runtime.apps.vlabench_big_small_brain import (
    VLABenchPilotConfig,
    run_vlabench_pilot,
)
from embodied_runtime.contracts import RobotObservation
from embodied_runtime.evaluation import ExperimentCondition
from embodied_runtime.simulators import EpisodeStep

np = pytest.importorskip("numpy")


@dataclass
class _Selected:
    action: object
    inference_latency_s: float = 0.001


class _Condition:
    def __init__(self, environment):
        self.environment = environment

    def is_met(self, physics):
        del physics
        return self.environment.placed


class _Task:
    def __init__(self, environment):
        self.conditions = type("Conditions", (), {})()
        self.conditions.conditions = [_Condition(environment)]


class _Inner:
    def __init__(self, environment):
        self.task = _Task(environment)
        self.physics = object()


class _RawEnvironment:
    def __init__(self, environment):
        self._env = _Inner(environment)


class _Endpoint:
    def __init__(self, task, *, max_episode_steps):
        del task, max_episode_steps
        self._raw_environment = _RawEnvironment(self)
        self.placed = False
        self.condition = None
        self.step_index = 0
        self.initial_fingerprint = "same-scene"

    @property
    def raw_environment(self):
        return self._raw_environment

    def reset(self, *, seed):
        del seed
        self.placed = False
        self.step_index = 0
        return RobotObservation(
            timestamp_s=1.0,
            values={"agent_pos": np.zeros(7), "pixels": {}},
        )

    def step(self, action):
        prompt = action.metadata["prompt"]
        self.step_index += 1
        if prompt.startswith("primitive: Pick"):
            self.placed = True
        success = self.placed and prompt.startswith("primitive: Press")
        return EpisodeStep(
            observation=RobotObservation(
                timestamp_s=float(self.step_index + 1),
                values={"agent_pos": np.zeros(7), "pixels": {}},
            ),
            reward=0.0,
            terminated=success,
            truncated=False,
            success=success,
            info={"is_success": success},
        )

    def close(self):
        pass


class _Runner:
    facts = None

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def load(self):
        return self

    def reset_action_queue(self):
        pass

    def select_action(self, observation, prompt):
        del observation, prompt
        return _Selected(np.zeros(7))


class _NeverCompletesEndpoint(_Endpoint):
    def step(self, action):
        del action
        self.step_index += 1
        return EpisodeStep(
            observation=RobotObservation(
                timestamp_s=float(self.step_index + 1),
                values={"agent_pos": np.zeros(7), "pixels": {}},
            ),
            reward=0.0,
            terminated=False,
            truncated=False,
            success=False,
            info={"is_success": False},
        )


class _PhysicsErrorEndpoint(_NeverCompletesEndpoint):
    def step(self, action):
        outcome = super().step(action)
        return EpisodeStep(
            observation=outcome.observation,
            reward=0.0,
            terminated=True,
            truncated=False,
            success=False,
            info={"is_success": False, "physics_error": True},
        )

    @property
    def raw_environment(self):
        raise AssertionError("terminal PhysicsError must not inspect the invalidated environment")


def test_pilot_produces_paired_report_and_persists_outputs(tmp_path):
    config = VLABenchPilotConfig(
        checkpoint="/checkpoint",
        output_dir=tmp_path,
        seeds=(4, 5),
        max_episode_steps=4,
        first_subgoal_budget=2,
    )

    report = run_vlabench_pilot(
        config,
        runner_factory=_Runner,
        endpoint_factory=_Endpoint,
        policy_seeder=lambda seed: None,
        clock=iter([0.0, 0.001] * 20).__next__,
    )

    by_condition = {summary.condition: summary for summary in report.summaries()}
    assert by_condition[ExperimentCondition.EDGE_ONLY].success_rate == 0.0
    assert by_condition[ExperimentCondition.ORACLE_PLAN].success_rate == 1.0
    assert by_condition[ExperimentCondition.ORACLE_PLAN].subgoal_completion_rate == 1.0
    assert report.comparisons()[0].effect_percentage_points == 100.0
    assert json.loads((tmp_path / "report.json").read_text())["oracle_gate"]["passed"] is False
    assert (tmp_path / "traces.jsonl").is_file()


def test_oracle_keeps_same_total_action_budget_as_edge_only(tmp_path):
    report = run_vlabench_pilot(
        VLABenchPilotConfig(
            checkpoint="/checkpoint",
            output_dir=tmp_path,
            seeds=(4,),
            max_episode_steps=4,
            first_subgoal_budget=2,
        ),
        runner_factory=_Runner,
        endpoint_factory=_NeverCompletesEndpoint,
        policy_seeder=lambda seed: None,
        clock=iter([0.0, 0.001] * 20).__next__,
    )

    assert {record.steps for record in report.records} == {4}


def test_physics_error_terminates_trial_without_inspecting_dropped_environment(tmp_path):
    report = run_vlabench_pilot(
        VLABenchPilotConfig(
            checkpoint="/checkpoint",
            output_dir=tmp_path,
            seeds=(4,),
            max_episode_steps=2,
            first_subgoal_budget=1,
        ),
        runner_factory=_Runner,
        endpoint_factory=_PhysicsErrorEndpoint,
        policy_seeder=lambda seed: None,
        clock=iter([0.0, 0.001] * 20).__next__,
    )

    assert all(record.steps == 1 and not record.success for record in report.records)


def test_config_rejects_nonpaired_or_out_of_scope_inputs(tmp_path):
    with pytest.raises(ValueError, match="duplicates"):
        VLABenchPilotConfig("/checkpoint", tmp_path, seeds=(1, 1))
    with pytest.raises(ValueError, match="only get_coffee"):
        VLABenchPilotConfig("/checkpoint", tmp_path, task="heat_food")
