from __future__ import annotations

from dataclasses import dataclass

import pytest

from embodied_runtime.robots.observation import RobotObservation
from embodied_runtime.simulators import EpisodeStep
from embodied_runtime.simulators.vlabench_privileged_oracle import (
    SIMULATOR_ORACLE_UPPER_BOUND,
    SIMULATOR_PRIVILEGED_PLAN_CONTROL,
    SIMULATOR_PRIVILEGED_SKILL_EXECUTOR,
    PrivilegedOracleTrajectory,
    PrivilegedOracleWaypoint,
    TexasHoldemDealIdentity,
    convert_vlabench_expert_waypoint,
    generate_texas_holdem_privileged_oracle_trajectory,
    generate_texas_holdem_privileged_skill_trajectory,
    replay_privileged_oracle_trajectory,
)

np = pytest.importorskip("numpy")


@dataclass
class _Poker:
    name: str


class _Task:
    def __init__(
        self,
        inner,
        *,
        cards=("2_of_clubs", "ace_of_hearts"),
        targets=("ace_of_hearts",),
    ):
        self._inner = inner
        self.pokers = [_Poker(name) for name in cards]
        self.entities = {poker.name: poker for poker in self.pokers}
        self._target_entities = {name: self.entities[name] for name in targets}
        self.max_cardtype = "high_card"

    @property
    def target_entities(self):
        return list(self._target_entities)

    def get_expert_skill_sequence(self, physics):
        assert physics is self._inner.physics
        selected_targets = tuple(self.target_entities)

        def privileged_skill(inner):
            assert inner is self._inner
            inner.privileged_raw_steps += 1
            inner.generated_for = selected_targets
            return (
                [],
                [
                    np.array([1.1, 2.2, 3.3, 0.1, 0.2, 0.3, 0.04, 0.04]),
                    np.array([1.2, 2.3, 3.4, 0.2, 0.3, 0.4, 0.0, 0.0]),
                ],
                True,
                True,
            )

        return [privileged_skill]

    def should_terminate_episode(self, physics):
        del physics
        return self._inner.privileged_raw_steps > 0


class _Inner:
    def __init__(self, *, cards, targets):
        self.physics = object()
        self.privileged_raw_steps = 0
        self.generated_for = ()
        self.task = _Task(self, cards=cards, targets=targets)

    @staticmethod
    def get_robot_frame_position():
        return np.array([0.1, 0.2, 0.3])


class _RawEnvironment:
    def __init__(self, *, cards, targets, expose_cached_base=True):
        self._env = _Inner(cards=cards, targets=targets)
        if expose_cached_base:
            self._robot_base_xyz = np.array([0.1, 0.2, 0.3])


class _Endpoint:
    def __init__(
        self,
        task="texas_holdem",
        *,
        max_episode_steps=2000,
        fingerprint="paired-fingerprint",
        cards=("2_of_clubs", "ace_of_hearts"),
        targets=("ace_of_hearts",),
        terminate_after=None,
    ):
        del task, max_episode_steps
        self.initial_fingerprint = fingerprint
        self.raw_environment = _RawEnvironment(cards=cards, targets=targets)
        self.terminate_after = terminate_after
        self.actions = []
        self.closed = False
        self.seed = None
        self._observation = RobotObservation(
            timestamp_s=1.0,
            values={"agent_pos": np.zeros(7), "pixels": {}},
            metadata={},
        )

    def reset(self, *, seed):
        self.seed = seed
        self._observation.metadata["seed"] = seed
        return self._observation

    def observe(self):
        return self._observation

    def step(self, action):
        self.actions.append(action)
        terminated = self.terminate_after is not None and len(self.actions) >= self.terminate_after
        return EpisodeStep(
            observation=self._observation,
            reward=0.0,
            terminated=terminated,
            truncated=False,
            success=terminated,
            info={"is_success": terminated},
        )

    def close(self):
        self.closed = True


def _trajectory(*, fingerprint="paired-fingerprint"):
    deal = TexasHoldemDealIdentity(
        card_names=("2_of_clubs", "ace_of_hearts"),
        target_card_names=("ace_of_hearts",),
        hand_type="high_card",
    )
    return PrivilegedOracleTrajectory(
        seed=1000,
        initial_fingerprint=fingerprint,
        deal=deal,
        selected_card_names=("ace_of_hearts",),
        waypoints=(
            PrivilegedOracleWaypoint(
                action=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0),
                skill_index=0,
                waypoint_index=0,
            ),
            PrivilegedOracleWaypoint(
                action=(0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0),
                skill_index=0,
                waypoint_index=1,
            ),
        ),
        shadow_task_success=True,
    )


def test_expert_waypoint_converts_world_frame_and_two_fingers_to_lerobot_7d():
    opened = convert_vlabench_expert_waypoint(
        [1.1, 2.2, 3.3, 0.1, 0.2, 0.3, 0.04, 0.04],
        robot_base_xyz=[0.1, 0.2, 0.3],
    )
    closed = convert_vlabench_expert_waypoint(
        [1.1, 2.2, 3.3, 0.1, 0.2, 0.3, 0.0, 0.0],
        robot_base_xyz=[0.1, 0.2, 0.3],
    )

    assert opened == pytest.approx((1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0))
    assert closed[-1] == 0.0


def test_expert_waypoint_rejects_ambiguous_finger_state():
    with pytest.raises(ValueError, match="finger positions disagree"):
        convert_vlabench_expert_waypoint(
            [1.1, 2.2, 3.3, 0.1, 0.2, 0.3, 0.04, 0.0],
            robot_base_xyz=[0.1, 0.2, 0.3],
        )


def test_shadow_generation_validates_pair_and_never_raw_steps_main_environment():
    main = _Endpoint()
    main.reset(seed=1000)
    shadows = []

    def shadow_factory(*args, **kwargs):
        shadow = _Endpoint(*args, **kwargs)
        shadows.append(shadow)
        return shadow

    trajectory = generate_texas_holdem_privileged_oracle_trajectory(
        main,
        seed=1000,
        target_card_names=("ace_of_hearts",),
        shadow_endpoint_factory=shadow_factory,
    )

    assert trajectory.oracle_label == SIMULATOR_ORACLE_UPPER_BOUND
    assert trajectory.is_oracle_upper_bound is True
    assert trajectory.deployable is False
    assert trajectory.shadow_task_success is True
    assert [waypoint.action[-1] for waypoint in trajectory.waypoints] == [1.0, 0.0]
    assert trajectory.waypoints[0].action[:3] == pytest.approx((1.0, 2.0, 3.0))
    assert main.raw_environment._env.privileged_raw_steps == 0
    assert shadows[0].raw_environment._env.privileged_raw_steps == 1
    assert shadows[0].raw_environment._env.generated_for == ("ace_of_hearts",)
    assert shadows[0].closed is True


def test_selected_wrong_card_controls_shadow_skills_without_becoming_oracle():
    main = _Endpoint()
    main.reset(seed=1000)
    shadows = []

    def shadow_factory(*args, **kwargs):
        shadow = _Endpoint(*args, **kwargs)
        shadows.append(shadow)
        return shadow

    trajectory = generate_texas_holdem_privileged_skill_trajectory(
        main,
        seed=1000,
        selected_card_names=("2_of_clubs",),
        shadow_endpoint_factory=shadow_factory,
    )

    assert trajectory.selected_card_names == ("2_of_clubs",)
    assert trajectory.is_oracle_upper_bound is False
    assert trajectory.evaluation_label == SIMULATOR_PRIVILEGED_PLAN_CONTROL
    assert "oracle" not in trajectory.evaluation_label
    assert shadows[0].raw_environment._env.generated_for == ("2_of_clubs",)


@pytest.mark.parametrize(
    ("shadow_fingerprint", "shadow_targets", "message"),
    [
        ("different", ("ace_of_hearts",), "fingerprint"),
        ("paired-fingerprint", ("2_of_clubs",), "poker deal"),
    ],
)
def test_shadow_generation_rejects_unpaired_scene_or_deal(
    shadow_fingerprint,
    shadow_targets,
    message,
):
    main = _Endpoint()
    main.reset(seed=1000)
    shadows = []

    def shadow_factory(*args, **kwargs):
        shadow = _Endpoint(
            *args,
            **kwargs,
            fingerprint=shadow_fingerprint,
            targets=shadow_targets,
        )
        shadows.append(shadow)
        return shadow

    with pytest.raises(RuntimeError, match=message):
        generate_texas_holdem_privileged_oracle_trajectory(
            main,
            seed=1000,
            target_card_names=("ace_of_hearts",),
            shadow_endpoint_factory=shadow_factory,
        )

    assert shadows[0].closed is True
    assert main.raw_environment._env.privileged_raw_steps == 0


def test_replay_uses_endpoint_step_and_adds_configurable_final_settle_repeats():
    main = _Endpoint()
    main.reset(seed=1000)

    replay = replay_privileged_oracle_trajectory(
        main,
        _trajectory(),
        settle_repeats=4,
        action_clock=lambda: 7.0,
    )

    assert replay.waypoint_steps == 2
    assert replay.settle_steps == 4
    assert replay.total_steps == 6
    assert main.raw_environment._env.privileged_raw_steps == 0
    assert len(main.actions) == 6
    assert all(action.timestamp_s == 7.0 for action in main.actions)
    assert all(
        action.metadata["controller"] == SIMULATOR_PRIVILEGED_SKILL_EXECUTOR
        and action.metadata["oracle_upper_bound"] is True
        and action.metadata["privileged_simulator_state"] is True
        and action.metadata["deployable"] is False
        for action in main.actions
    )
    assert [action.metadata["phase"] for action in main.actions] == [
        "expert_waypoint",
        "expert_waypoint",
        "final_settle",
        "final_settle",
        "final_settle",
        "final_settle",
    ]
    assert all(
        action.values["action"] == _trajectory().waypoints[-1].action for action in main.actions[2:]
    )


def test_replay_stops_at_terminal_transition_without_settle():
    main = _Endpoint(terminate_after=1)
    main.reset(seed=1000)

    replay = replay_privileged_oracle_trajectory(
        main,
        _trajectory(),
        settle_repeats=10,
    )

    assert replay.success is True
    assert replay.waypoint_steps == 1
    assert replay.settle_steps == 0
    assert len(main.actions) == 1
