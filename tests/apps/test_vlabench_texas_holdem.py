from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from embodied_runtime.apps.vlabench_texas_holdem import (
    TEXAS_HOLDEM_COMPOSITE_PROMPT,
    EndpointPlaceControllerConfig,
    PokerCard,
    PokerDeal,
    TexasHoldemExperimentConfig,
    _rotated_conditions,
    _wrong_plan_cards,
    run_texas_holdem_experiment,
)
from embodied_runtime.contracts import RobotObservation
from embodied_runtime.evaluation import ExperimentCondition
from embodied_runtime.simulators import EpisodeStep

np = pytest.importorskip("numpy")


@dataclass
class _Selected:
    action: object
    inference_latency_s: float = 0.001
    generated_chunk: bool = True
    queue_remaining: int = 0


class _Card:
    def __init__(self, endpoint, name, value, suit):
        self.endpoint = endpoint
        self.name = name
        self.value = value
        self.suite = suit

    def is_grasped(self, physics, robot):
        del physics, robot
        return self.endpoint.held == self.name

    def get_xpos(self, physics):
        del physics
        if self.name in self.endpoint.placed:
            return np.array([0.0, 0.0, 0.8])
        return np.array([3.0, 0.0, 0.8])


class _Placemat:
    def get_place_point(self, physics):
        del physics
        return [np.array([0.0, 0.0, 0.8])]

    def contain(self, point, physics):
        del physics
        return abs(float(point[0])) < 0.5


class _Robot:
    @staticmethod
    def ee_offset(physics):
        del physics
        return np.zeros(3)


class _Task:
    def __init__(self, endpoint):
        self.robot = _Robot()
        self.pokers = [
            _Card(endpoint, "2_of_clubs", "2", "clubs"),
            _Card(endpoint, "ace_of_hearts", "ace", "hearts"),
        ]
        self.target_entities = ["ace_of_hearts"]
        self.max_cardtype = "high_card"
        self.entities = {card.name: card for card in self.pokers}
        self.entities["target_container"] = _Placemat()


class _Inner:
    def __init__(self, endpoint):
        self.task = _Task(endpoint)
        self.physics = object()


class _RawEnvironment:
    def __init__(self, endpoint):
        self._env = _Inner(endpoint)
        self._robot_base_xyz = np.zeros(3)


class _Endpoint:
    def __init__(self, task, *, max_episode_steps):
        del task, max_episode_steps
        self._raw_environment = _RawEnvironment(self)
        self.initial_fingerprint = "same-scene"
        self.held = None
        self.placed = set()
        self.step_index = 0
        self.history = []

    @property
    def raw_environment(self):
        return self._raw_environment

    def reset(self, *, seed):
        del seed
        self.held = None
        self.placed = set()
        self.step_index = 0
        return self._observation()

    def _observation(self):
        return RobotObservation(
            timestamp_s=float(self.step_index + 1),
            values={
                "agent_pos": np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
                "pixels": {},
            },
        )

    def step(self, action):
        self.history.append(action)
        self.step_index += 1
        controller = action.metadata["controller"]
        if controller == "edge_policy":
            prompt = action.metadata["prompt"]
            if prompt == "primitive: Please pick the poker ace of hearts":
                self.held = "ace_of_hearts"
            elif prompt == "primitive: Please pick the poker 2 of clubs" or (
                prompt == TEXAS_HOLDEM_COMPOSITE_PROMPT and "2_of_clubs" not in self.placed
            ):
                self.held = "2_of_clubs"
        elif controller == "shared_endpoint_place":
            action_values = np.asarray(action.values["action"])
            if action_values[-1] >= 0.5 and self.held is not None:
                self.placed.add(self.held)
                self.held = None

        success = "ace_of_hearts" in self.placed
        return EpisodeStep(
            observation=self._observation(),
            reward=0.0,
            terminated=success,
            truncated=False,
            success=success,
            info={"is_success": success},
        )

    def close(self):
        pass


class _NeverGraspsEndpoint(_Endpoint):
    def step(self, action):
        self.history.append(action)
        self.step_index += 1
        return EpisodeStep(
            observation=self._observation(),
            reward=0.0,
            terminated=False,
            truncated=False,
            success=False,
            info={"is_success": False},
        )


class _Runner:
    facts = None

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.reset_count = 0

    def load(self):
        return self

    def reset_action_queue(self):
        self.reset_count += 1

    def select_action(self, observation, prompt):
        del observation, prompt
        return _Selected(np.zeros(7))


def _clock():
    value = -0.001

    def now():
        nonlocal value
        value += 0.001
        return value

    return now


def _fast_controller():
    return EndpointPlaceControllerConfig(
        lift_height_m=0.01,
        clearance_m=0.01,
        retract_height_m=0.01,
        max_translation_step_m=10.0,
        open_steps=1,
        slot_count=1,
        slot_spacing_m=0.01,
    )


def test_texas_holdem_paired_experiment_isolates_oracle_card_selection(tmp_path):
    endpoints = []

    def endpoint_factory(*args, **kwargs):
        endpoint = _Endpoint(*args, **kwargs)
        endpoints.append(endpoint)
        return endpoint

    report = run_texas_holdem_experiment(
        TexasHoldemExperimentConfig(
            checkpoint="/checkpoint",
            output_dir=tmp_path,
            seeds=(7,),
            max_episode_steps=12,
            warmup_policy=False,
            place_controller=_fast_controller(),
        ),
        runner_factory=_Runner,
        endpoint_factory=endpoint_factory,
        policy_seeder=lambda seed: None,
        clock=_clock(),
    )

    by_condition = {record.condition: record for record in report.records}
    assert not by_condition[ExperimentCondition.EDGE_ONLY].success
    assert by_condition[ExperimentCondition.ORACLE_PLAN].success
    assert not by_condition[ExperimentCondition.WRONG_PLAN].success
    assert by_condition[ExperimentCondition.ORACLE_PLAN].subgoals_completed == 1
    assert by_condition[ExperimentCondition.ORACLE_PLAN].subgoals_total == 1
    assert len(by_condition[ExperimentCondition.ORACLE_PLAN].planner_latencies_ms) == 1
    assert len(by_condition[ExperimentCondition.WRONG_PLAN].planner_latencies_ms) == 1
    assert report.comparisons()[0].effect_percentage_points == 100.0

    controller_actions = [
        action
        for action in endpoints[0].history
        if action.metadata["controller"] == "shared_endpoint_place"
    ]
    assert {action.metadata["condition"] for action in controller_actions} == {
        ExperimentCondition.EDGE_ONLY.value,
        ExperimentCondition.ORACLE_PLAN.value,
        ExperimentCondition.WRONG_PLAN.value,
    }
    assert all(
        action.metadata["controller"] == "shared_endpoint_place" for action in controller_actions
    )
    planned_policy_actions = [
        action
        for action in endpoints[0].history
        if action.metadata["controller"] == "edge_policy"
        and action.metadata["condition"]
        in {
            ExperimentCondition.ORACLE_PLAN.value,
            ExperimentCondition.WRONG_PLAN.value,
        }
    ]
    assert {action.metadata["condition"] for action in planned_policy_actions} == {
        ExperimentCondition.ORACLE_PLAN.value,
        ExperimentCondition.WRONG_PLAN.value,
    }
    assert all(
        action.metadata["prompt"].startswith("primitive: Please pick the poker ")
        for action in planned_policy_actions
    )

    traces = [json.loads(line) for line in (tmp_path / "traces.jsonl").read_text().splitlines()]
    oracle_trace = next(
        trace for trace in traces if trace["condition"] == ExperimentCondition.ORACLE_PLAN.value
    )
    wrong_trace = next(
        trace for trace in traces if trace["condition"] == ExperimentCondition.WRONG_PLAN.value
    )
    assert oracle_trace["deal"]["target_prompts"] == [
        "primitive: Please pick the poker ace of hearts"
    ]
    wrong_plan = next(
        event for event in wrong_trace["events"] if event["event"] == "plan_activated"
    )
    assert wrong_plan["planned_card_names"] == ["2_of_clubs"]
    assert wrong_plan["deliberately_incorrect"] is True
    assert {trace["initial_fingerprint"] for trace in traces} == {"same-scene"}
    assert "upper bound" in json.loads((tmp_path / "run_config.json").read_text())["oracle_caveat"]
    control = json.loads((tmp_path / "plan_quality_control.json").read_text())
    assert control["comparison"]["baseline"] == ExperimentCondition.WRONG_PLAN.value
    assert control["comparison"]["comparator"] == ExperimentCondition.ORACLE_PLAN.value
    assert control["comparison"]["effect_percentage_points"] == 100.0


def test_all_conditions_receive_the_same_total_action_budget(tmp_path):
    report = run_texas_holdem_experiment(
        TexasHoldemExperimentConfig(
            checkpoint="/checkpoint",
            output_dir=tmp_path,
            seeds=(8,),
            max_episode_steps=4,
            warmup_policy=False,
            place_controller=_fast_controller(),
        ),
        runner_factory=_Runner,
        endpoint_factory=_NeverGraspsEndpoint,
        policy_seeder=lambda seed: None,
        clock=_clock(),
    )

    assert {record.steps for record in report.records} == {4}
    assert all(not record.success for record in report.records)


def test_wrong_plan_is_seeded_equal_length_and_replaces_exactly_one_target():
    cards = tuple(
        PokerCard(name, value, suit)
        for name, value, suit in (
            ("2_of_clubs", "2", "clubs"),
            ("3_of_hearts", "3", "hearts"),
            ("4_of_diamonds", "4", "diamonds"),
            ("5_of_spades", "5", "spades"),
        )
    )
    deal = PokerDeal(
        cards=cards,
        target_names=("3_of_hearts", "4_of_diamonds"),
        hand_type="one_pair",
    )

    first = _wrong_plan_cards(deal, seed=41)
    repeated = _wrong_plan_cards(deal, seed=41)

    assert first == repeated
    assert len(first) == len(deal.targets)
    selected_names = {card.name for card in first}
    target_names = set(deal.target_names)
    assert len(selected_names - target_names) == 1
    assert len(target_names - selected_names) == 1


def test_three_condition_order_is_counterbalanced_by_rotation():
    assert _rotated_conditions(0) == (
        ExperimentCondition.EDGE_ONLY,
        ExperimentCondition.ORACLE_PLAN,
        ExperimentCondition.WRONG_PLAN,
    )
    assert _rotated_conditions(1) == (
        ExperimentCondition.ORACLE_PLAN,
        ExperimentCondition.WRONG_PLAN,
        ExperimentCondition.EDGE_ONLY,
    )
    assert _rotated_conditions(2) == (
        ExperimentCondition.WRONG_PLAN,
        ExperimentCondition.EDGE_ONLY,
        ExperimentCondition.ORACLE_PLAN,
    )


def test_exact_training_prompts_and_configuration_scope(tmp_path):
    card = PokerCard("queen_of_spades", "queen", "spades")
    assert card.primitive_prompt == "primitive: Please pick the poker queen of spades"
    assert TEXAS_HOLDEM_COMPOSITE_PROMPT == (
        "composite: We're playing Texas hodl'em game! "
        "What's your largest cards? Show me on the placemat"
    )

    with pytest.raises(ValueError, match="duplicates"):
        TexasHoldemExperimentConfig("/checkpoint", tmp_path, seeds=(1, 1))
    with pytest.raises(ValueError, match="only the texas_holdem"):
        TexasHoldemExperimentConfig("/checkpoint", tmp_path, task="play_poker")
