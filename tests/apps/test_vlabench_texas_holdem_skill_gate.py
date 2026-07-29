from __future__ import annotations

import json
import time
from types import SimpleNamespace

from embodied_runtime.apps.vlabench_texas_holdem_skill_gate import (
    PlannerPokerDeal,
    PlannerSkillCondition,
    PlannerSkillGateConfig,
    build_cloud_plan_request,
    deterministic_wrong_card_selection,
    run_planner_skill_gate,
)
from embodied_runtime.contracts import PlanEnvelope, PlanStep, RobotObservation
from embodied_runtime.integrations.planning import TexasHoldemCard

_DEAL = PlannerPokerDeal(
    cards=(
        TexasHoldemCard("3_of_spades", "3", "spades"),
        TexasHoldemCard("9_of_clubs", "9", "clubs"),
        TexasHoldemCard("4_of_diamonds", "4", "diamonds"),
        TexasHoldemCard("10_of_diamonds", "10", "diamonds"),
        TexasHoldemCard("8_of_hearts", "8", "hearts"),
        TexasHoldemCard("4_of_spades", "4", "spades"),
        TexasHoldemCard("5_of_diamonds", "5", "diamonds"),
    ),
    target_card_names=("4_of_diamonds", "4_of_spades"),
    hand_type="one_pair",
)


class _Endpoint:
    initial_fingerprint = "same-reset"

    def __init__(self, task, *, max_episode_steps, render_resolution):
        del task, max_episode_steps, render_resolution
        self.closed = False

    def reset(self, *, seed):
        return RobotObservation(
            timestamp_s=float(seed + 1),
            values={"agent_pos": [0.0] * 7},
        )

    def close(self):
        self.closed = True


class _CloudPlanner:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.requests = []

    def plan(self, request):
        self.requests.append(request)
        if self.fail:
            raise ValueError("synthetic invalid planner output")
        now = time.time()
        return PlanEnvelope(
            request_id=request.request_id,
            task_id=request.goal.task_id,
            session_id=request.goal.session_id,
            revision=1,
            steps=tuple(
                PlanStep(
                    step_id=f"pick_{index}",
                    instruction=f"pick {name}",
                    skill="pick_and_place_poker",
                    metadata={"poker_name": name},
                )
                for index, name in enumerate(_DEAL.target_card_names)
            ),
            created_at_s=now,
            expires_at_s=now + 60.0,
            based_on_observation_id=request.observation_id,
        )


def _skill_executor(endpoint, *, selected_card_names, **kwargs):
    del endpoint, kwargs
    selected = tuple(selected_card_names)
    success = selected == _DEAL.target_card_names
    return SimpleNamespace(
        success=success,
        waypoint_steps=20,
        settle_steps=0 if success else 3,
        trajectory=SimpleNamespace(
            evaluation_label=(
                "simulator_oracle_privileged_upper_bound"
                if success
                else "simulator_privileged_plan_control"
            )
        ),
    )


def test_gate_holds_executor_fixed_for_wrong_oracle_and_cloud(tmp_path):
    planner = _CloudPlanner()
    trials = run_planner_skill_gate(
        PlannerSkillGateConfig(output_dir=tmp_path, seeds=(1000,)),
        cloud_planner=planner,
        endpoint_factory=_Endpoint,
        deal_inspector=lambda endpoint: _DEAL,
        skill_executor=_skill_executor,
        clock=iter((1.0, 1.25)).__next__,
    )

    by_condition = {trial.condition: trial for trial in trials}
    assert set(by_condition) == set(PlannerSkillCondition)
    assert not by_condition[PlannerSkillCondition.WRONG_PLAN].success
    assert by_condition[PlannerSkillCondition.ORACLE_PLAN].success
    assert by_condition[PlannerSkillCondition.CLOUD_PLAN].success
    assert (
        by_condition[PlannerSkillCondition.WRONG_PLAN].selected_card_names
        != _DEAL.target_card_names
    )
    assert all(trial.execution_attempted for trial in trials)
    assert planner.requests[0].goal.metadata["texas_holdem"]["cards"][2] == {
        "name": "4_of_diamonds",
        "value": "4",
        "suit": "diamonds",
    }

    report = json.loads((tmp_path / "planner_skill_gate.json").read_text())
    oracle_control = next(
        comparison
        for comparison in report["comparisons"]
        if comparison["comparator"] == "oracle_plan"
    )
    assert oracle_control["effect_percentage_points"] == 100.0
    assert "non-deployable" in report["claim_scope"]


def test_invalid_cloud_plan_is_rejected_without_robot_execution(tmp_path):
    selected_by_executor = []

    def executor(endpoint, *, selected_card_names, **kwargs):
        selected_by_executor.append(tuple(selected_card_names))
        return _skill_executor(
            endpoint,
            selected_card_names=selected_card_names,
            **kwargs,
        )

    trials = run_planner_skill_gate(
        PlannerSkillGateConfig(output_dir=tmp_path, seeds=(7,)),
        cloud_planner=_CloudPlanner(fail=True),
        endpoint_factory=_Endpoint,
        deal_inspector=lambda endpoint: _DEAL,
        skill_executor=executor,
        clock=iter((2.0, 2.1)).__next__,
    )

    cloud = next(trial for trial in trials if trial.condition is PlannerSkillCondition.CLOUD_PLAN)
    assert not cloud.planner_accepted
    assert not cloud.execution_attempted
    assert cloud.error_type == "ValueError"
    assert len(selected_by_executor) == 2


def test_cloud_request_has_lineage_but_does_not_leak_oracle_targets():
    request = build_cloud_plan_request(
        deal=_DEAL,
        seed=1000,
        fingerprint="frame-1",
        observation_timestamp_s=10.0,
    )

    assert request.observation is None
    assert request.observation_id == "frame-1"
    assert "target_card_names" not in json.dumps(request.goal.metadata)
    assert request.goal.allowed_skills == ("pick_and_place_poker",)


def test_wrong_selection_is_reproducible_and_preserves_plan_length():
    first = deterministic_wrong_card_selection(_DEAL, seed=1000)
    second = deterministic_wrong_card_selection(_DEAL, seed=1000)

    assert first == second
    assert len(first) == len(_DEAL.target_card_names)
    assert set(first) != set(_DEAL.target_card_names)
    assert len(set(first) & set(_DEAL.target_card_names)) == len(first) - 1
