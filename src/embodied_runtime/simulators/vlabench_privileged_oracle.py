"""Privileged VLABench skill bridge for simulator-only controlled trials.

This module is deliberately *not* a deployable policy implementation.
VLABench's expert skills inspect simulator state and advance a private shadow
environment while producing waypoints.  The resulting trajectory can then be
replayed on the paired main environment, but every main-environment transition
must pass through :class:`~embodied_runtime.simulators.SimulatorEndpoint`.

The executor is privileged regardless of who selected the cards.  A trajectory
is an oracle upper bound only when its selected cards exactly equal the
simulator's ground-truth best hand.  A wrong or cloud-selected plan remains a
privileged-executor control and must never be reported as an oracle result or
an edge/cloud deployment result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from embodied_runtime.contracts import RobotAction

from .base import EpisodeStep
from .vlabench import VLABenchSimulatorEndpoint

TEXAS_HOLDEM_TASK = "texas_holdem"
SIMULATOR_PRIVILEGED_SKILL_EXECUTOR = "simulator_privileged_skill_executor"
SIMULATOR_ORACLE_UPPER_BOUND = "simulator_oracle_privileged_upper_bound"
SIMULATOR_PRIVILEGED_PLAN_CONTROL = "simulator_privileged_plan_control"
_DEFAULT_GRIPPER_OPEN_THRESHOLD = 0.03
_DEFAULT_FINGER_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class TexasHoldemDealIdentity:
    """Stable simulator-visible identity used to validate a paired deal."""

    card_names: tuple[str, ...]
    target_card_names: tuple[str, ...]
    hand_type: str

    def __post_init__(self) -> None:
        card_names = _non_empty_unique_names(self.card_names, field_name="card_names")
        target_card_names = _non_empty_unique_names(
            self.target_card_names,
            field_name="target_card_names",
        )
        missing = set(target_card_names).difference(card_names)
        if missing:
            raise ValueError(f"target cards are missing from the deal: {sorted(missing)}")
        if not isinstance(self.hand_type, str) or not self.hand_type.strip():
            raise ValueError("hand_type must be a non-empty string")
        object.__setattr__(self, "card_names", card_names)
        object.__setattr__(self, "target_card_names", target_card_names)
        object.__setattr__(self, "hand_type", self.hand_type.strip())


@dataclass(frozen=True, slots=True)
class PrivilegedSkillWaypoint:
    """One deployable-shaped action generated from a privileged 8D waypoint."""

    action: tuple[float, float, float, float, float, float, float]
    skill_index: int
    waypoint_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, tuple) or len(self.action) != 7:
            raise ValueError("privileged skill waypoint action must be a 7D tuple")
        if any(not math.isfinite(value) for value in self.action):
            raise ValueError("privileged skill waypoint action must contain only finite values")
        for field_name in ("skill_index", "waypoint_index"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True, slots=True)
class PrivilegedSkillTrajectory:
    """Privileged skill trajectory that is explicitly ineligible for deployment."""

    seed: int
    initial_fingerprint: str
    deal: TexasHoldemDealIdentity
    selected_card_names: tuple[str, ...]
    waypoints: tuple[PrivilegedSkillWaypoint, ...]
    shadow_task_success: bool
    executor_label: str = SIMULATOR_PRIVILEGED_SKILL_EXECUTOR
    deployable: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("privileged skill trajectory seed must be an integer")
        if self.seed < 0:
            raise ValueError("privileged skill trajectory seed must be non-negative")
        if not isinstance(self.initial_fingerprint, str) or not self.initial_fingerprint.strip():
            raise ValueError("initial_fingerprint must be a non-empty string")
        if not isinstance(self.deal, TexasHoldemDealIdentity):
            raise TypeError("deal must be a TexasHoldemDealIdentity")
        selected_card_names = _non_empty_unique_names(
            self.selected_card_names,
            field_name="selected_card_names",
        )
        missing = set(selected_card_names).difference(self.deal.card_names)
        if missing:
            raise ValueError(f"selected cards are missing from the deal: {sorted(missing)}")
        waypoints = tuple(self.waypoints)
        if not waypoints:
            raise ValueError("privileged skill trajectory must contain waypoints")
        if any(not isinstance(waypoint, PrivilegedSkillWaypoint) for waypoint in waypoints):
            raise TypeError("waypoints must contain PrivilegedSkillWaypoint values")
        if not isinstance(self.shadow_task_success, bool):
            raise TypeError("shadow_task_success must be a bool")
        if self.executor_label != SIMULATOR_PRIVILEGED_SKILL_EXECUTOR:
            raise ValueError(
                f"executor_label must remain {SIMULATOR_PRIVILEGED_SKILL_EXECUTOR!r} "
                "so privileged results cannot be mislabeled"
            )
        if self.deployable is not False:
            raise ValueError("a privileged simulator-skill trajectory cannot be deployable")
        object.__setattr__(self, "initial_fingerprint", self.initial_fingerprint.strip())
        object.__setattr__(self, "selected_card_names", selected_card_names)
        object.__setattr__(self, "waypoints", waypoints)

    @property
    def is_oracle_upper_bound(self) -> bool:
        return frozenset(self.selected_card_names) == frozenset(self.deal.target_card_names)

    @property
    def evaluation_label(self) -> str:
        if self.is_oracle_upper_bound:
            return SIMULATOR_ORACLE_UPPER_BOUND
        return SIMULATOR_PRIVILEGED_PLAN_CONTROL

    @property
    def oracle_label(self) -> str:
        """Backward-readable label; only contains ``oracle`` for the true plan."""

        return self.evaluation_label


@dataclass(frozen=True, slots=True)
class PrivilegedSkillReplay:
    """Compact result of endpoint-only replay on the main environment."""

    trajectory: PrivilegedSkillTrajectory
    waypoint_steps: int
    settle_steps: int
    success: bool
    terminated: bool
    truncated: bool
    final_outcome: EpisodeStep | None

    @property
    def total_steps(self) -> int:
        return self.waypoint_steps + self.settle_steps

    @property
    def evaluation_label(self) -> str:
        return self.trajectory.evaluation_label


def convert_vlabench_expert_waypoint(
    waypoint: Any,
    *,
    robot_base_xyz: Any,
    gripper_open_threshold: float = _DEFAULT_GRIPPER_OPEN_THRESHOLD,
    finger_tolerance: float = _DEFAULT_FINGER_TOLERANCE,
) -> tuple[float, float, float, float, float, float, float]:
    """Convert VLABench expert ``xyz_world+euler+two_fingers`` to LeRobot 7D.

    The returned position is relative to the robot base, matching LeRobot's
    VLABench dataset/action convention.  The two equal finger positions are
    collapsed to one binary gripper value: values above ``0.03`` are open.
    """

    np = _numpy()
    raw = np.asarray(waypoint, dtype=np.float64).reshape(-1)
    if raw.shape != (8,):
        raise ValueError(
            "VLABench expert waypoint must have shape (8,) "
            "[xyz_world, euler_xyz, left_finger, right_finger]"
        )
    base = np.asarray(robot_base_xyz, dtype=np.float64).reshape(-1)
    if base.shape != (3,):
        raise ValueError("robot_base_xyz must have shape (3,)")
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(base)):
        raise ValueError("waypoint and robot_base_xyz must contain only finite values")
    threshold = _finite_non_negative(
        gripper_open_threshold,
        field_name="gripper_open_threshold",
    )
    tolerance = _finite_non_negative(finger_tolerance, field_name="finger_tolerance")
    if abs(float(raw[6]) - float(raw[7])) > tolerance:
        raise ValueError(
            "VLABench expert waypoint finger positions disagree; "
            "cannot safely collapse two fingers to one binary command"
        )

    action = np.empty(7, dtype=np.float64)
    action[:3] = raw[:3] - base
    action[3:6] = raw[3:6]
    action[6] = float(float(raw[6]) > threshold)
    return tuple(float(value) for value in action)  # type: ignore[return-value]


def generate_texas_holdem_privileged_skill_trajectory(
    main_endpoint: Any,
    *,
    seed: int,
    selected_card_names: Sequence[str],
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
) -> PrivilegedSkillTrajectory:
    """Generate selected-card expert waypoints in an identically seeded shadow env.

    ``main_endpoint`` must already be reset to ``seed``.  Before any skill is
    called, the bridge verifies the selected cards and then requires
    the shadow environment to reproduce both the main observation fingerprint
    and the full poker deal.  Expert skills are invoked only on the shadow
    environment's inner VLABench object.
    """

    _validate_seed(seed)
    selected_cards = _non_empty_unique_names(
        selected_card_names,
        field_name="selected_card_names",
    )
    if (
        isinstance(shadow_max_episode_steps, bool)
        or not isinstance(shadow_max_episode_steps, int)
        or shadow_max_episode_steps <= 0
    ):
        raise ValueError("shadow_max_episode_steps must be a positive integer")
    if shadow_endpoint_factory is not None and not callable(shadow_endpoint_factory):
        raise TypeError("shadow_endpoint_factory must be callable or None")

    main_fingerprint = _endpoint_fingerprint(main_endpoint)
    main_deal = inspect_texas_holdem_deal(main_endpoint)
    _validate_selected_cards(selected_cards, main_deal)
    _validate_endpoint_seed_if_available(main_endpoint, seed)

    factory = shadow_endpoint_factory or VLABenchSimulatorEndpoint
    shadow_endpoint = factory(
        TEXAS_HOLDEM_TASK,
        max_episode_steps=shadow_max_episode_steps,
    )
    try:
        shadow_endpoint.reset(seed=seed)
        shadow_fingerprint = _endpoint_fingerprint(shadow_endpoint)
        shadow_deal = inspect_texas_holdem_deal(shadow_endpoint)
        _validate_endpoint_seed_if_available(shadow_endpoint, seed)
        if shadow_fingerprint != main_fingerprint:
            raise RuntimeError(
                "privileged oracle shadow environment did not reproduce the main "
                "initial observation fingerprint"
            )
        if shadow_deal != main_deal:
            raise RuntimeError(
                "privileged oracle shadow environment did not reproduce the main poker deal"
            )

        shadow_raw = shadow_endpoint.raw_environment
        shadow_inner = getattr(shadow_raw, "_env", None)
        shadow_task = getattr(shadow_inner, "task", None)
        shadow_physics = getattr(shadow_inner, "physics", None)
        if shadow_inner is None or shadow_task is None or shadow_physics is None:
            raise RuntimeError("shadow endpoint did not expose a live VLABench task and physics")
        get_sequence = getattr(shadow_task, "get_expert_skill_sequence", None)
        if not callable(get_sequence):
            raise TypeError("Texas Hold'em task did not expose get_expert_skill_sequence()")
        skills = _get_selected_expert_skill_sequence(
            shadow_task,
            shadow_physics,
            selected_cards,
        )
        if not isinstance(skills, Sequence) or isinstance(skills, (str, bytes)) or not skills:
            raise RuntimeError("Texas Hold'em expert skill sequence is empty or invalid")

        robot_base_xyz = _robot_base_xyz(shadow_raw, shadow_inner)
        converted: list[PrivilegedSkillWaypoint] = []
        shadow_task_success = False
        for skill_index, skill in enumerate(skills):
            if not callable(skill):
                raise TypeError(f"expert skill at index {skill_index} is not callable")
            result = skill(shadow_inner)
            if result is None or not isinstance(result, Sequence) or len(result) < 2:
                raise RuntimeError(f"expert skill at index {skill_index} did not return waypoints")
            raw_waypoints = result[1]
            if raw_waypoints is None:
                raise RuntimeError(f"expert skill at index {skill_index} returned no waypoints")
            for waypoint_index, waypoint in enumerate(raw_waypoints):
                converted.append(
                    PrivilegedSkillWaypoint(
                        action=convert_vlabench_expert_waypoint(
                            waypoint,
                            robot_base_xyz=robot_base_xyz,
                        ),
                        skill_index=skill_index,
                        waypoint_index=waypoint_index,
                    )
                )
            if len(result) >= 4 and bool(result[3]):
                shadow_task_success = True
                break

        if not shadow_task_success:
            should_terminate = getattr(shadow_task, "should_terminate_episode", None)
            if callable(should_terminate):
                shadow_task_success = bool(should_terminate(shadow_physics))
        return PrivilegedSkillTrajectory(
            seed=seed,
            initial_fingerprint=main_fingerprint,
            deal=main_deal,
            selected_card_names=selected_cards,
            waypoints=tuple(converted),
            shadow_task_success=shadow_task_success,
        )
    finally:
        close = getattr(shadow_endpoint, "close", None)
        if callable(close):
            close()


def replay_privileged_skill_trajectory(
    main_endpoint: Any,
    trajectory: PrivilegedSkillTrajectory,
    *,
    settle_repeats: int = 3,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Replay privileged waypoints through ``main_endpoint.step`` only.

    Repeating the final waypoint is configurable because VLABench's IK replay
    can stop roughly one centimetre short after a single control update.
    """

    if not isinstance(trajectory, PrivilegedSkillTrajectory):
        raise TypeError("trajectory must be a PrivilegedSkillTrajectory")
    if isinstance(settle_repeats, bool) or not isinstance(settle_repeats, int):
        raise TypeError("settle_repeats must be an integer")
    if settle_repeats < 0:
        raise ValueError("settle_repeats must be non-negative")
    if not callable(action_clock):
        raise TypeError("action_clock must be callable")

    current_fingerprint = _endpoint_fingerprint(main_endpoint)
    if current_fingerprint != trajectory.initial_fingerprint:
        raise RuntimeError("main endpoint fingerprint changed before privileged oracle replay")
    if inspect_texas_holdem_deal(main_endpoint) != trajectory.deal:
        raise RuntimeError("main endpoint poker deal changed before privileged oracle replay")
    _validate_endpoint_seed_if_available(main_endpoint, trajectory.seed)

    final_outcome: EpisodeStep | None = None
    waypoint_steps = 0
    settle_steps = 0
    for waypoint in trajectory.waypoints:
        final_outcome = _step_main_endpoint(
            main_endpoint,
            waypoint,
            phase="expert_waypoint",
            repeat_index=None,
            evaluation_label=trajectory.evaluation_label,
            is_oracle_upper_bound=trajectory.is_oracle_upper_bound,
            selected_card_names=trajectory.selected_card_names,
            action_clock=action_clock,
        )
        waypoint_steps += 1
        if final_outcome.done:
            break

    if final_outcome is None or not final_outcome.done:
        final_waypoint = trajectory.waypoints[-1]
        for repeat_index in range(settle_repeats):
            final_outcome = _step_main_endpoint(
                main_endpoint,
                final_waypoint,
                phase="final_settle",
                repeat_index=repeat_index,
                evaluation_label=trajectory.evaluation_label,
                is_oracle_upper_bound=trajectory.is_oracle_upper_bound,
                selected_card_names=trajectory.selected_card_names,
                action_clock=action_clock,
            )
            settle_steps += 1
            if final_outcome.done:
                break

    return PrivilegedSkillReplay(
        trajectory=trajectory,
        waypoint_steps=waypoint_steps,
        settle_steps=settle_steps,
        success=bool(final_outcome.success) if final_outcome is not None else False,
        terminated=bool(final_outcome.terminated) if final_outcome is not None else False,
        truncated=bool(final_outcome.truncated) if final_outcome is not None else False,
        final_outcome=final_outcome,
    )


def run_texas_holdem_privileged_skill_executor(
    main_endpoint: Any,
    *,
    seed: int,
    selected_card_names: Sequence[str],
    settle_repeats: int = 3,
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Generate selected-card skills in shadow and replay them on main."""

    trajectory = generate_texas_holdem_privileged_skill_trajectory(
        main_endpoint,
        seed=seed,
        selected_card_names=selected_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    return replay_privileged_skill_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


def generate_texas_holdem_privileged_oracle_trajectory(
    main_endpoint: Any,
    *,
    seed: int,
    target_card_names: Sequence[str],
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
) -> PrivilegedSkillTrajectory:
    """Backward-compatible strict oracle entry point.

    Unlike the general privileged skill executor, this function rejects any
    selection that is not exactly the simulator's ground-truth best hand.
    """

    trajectory = generate_texas_holdem_privileged_skill_trajectory(
        main_endpoint,
        seed=seed,
        selected_card_names=target_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    if not trajectory.is_oracle_upper_bound:
        raise RuntimeError(
            "privileged oracle entry point requires all and only the true target cards"
        )
    return trajectory


def replay_privileged_oracle_trajectory(
    main_endpoint: Any,
    trajectory: PrivilegedSkillTrajectory,
    *,
    settle_repeats: int = 3,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Backward-compatible replay alias for a strict oracle trajectory."""

    if not isinstance(trajectory, PrivilegedSkillTrajectory):
        raise TypeError("trajectory must be a PrivilegedSkillTrajectory")
    if not trajectory.is_oracle_upper_bound:
        raise RuntimeError("oracle replay cannot label a non-oracle selected-card plan")
    return replay_privileged_skill_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


def run_texas_holdem_privileged_oracle_upper_bound(
    main_endpoint: Any,
    *,
    seed: int,
    target_card_names: Sequence[str],
    settle_repeats: int = 3,
    shadow_endpoint_factory: Callable[..., Any] | None = None,
    shadow_max_episode_steps: int = 2000,
    action_clock: Callable[[], float] = time.time,
) -> PrivilegedSkillReplay:
    """Strict oracle upper bound using the general privileged skill executor."""

    trajectory = generate_texas_holdem_privileged_oracle_trajectory(
        main_endpoint,
        seed=seed,
        target_card_names=target_card_names,
        shadow_endpoint_factory=shadow_endpoint_factory,
        shadow_max_episode_steps=shadow_max_episode_steps,
    )
    return replay_privileged_oracle_trajectory(
        main_endpoint,
        trajectory,
        settle_repeats=settle_repeats,
        action_clock=action_clock,
    )


def inspect_texas_holdem_deal(endpoint: Any) -> TexasHoldemDealIdentity:
    """Inspect privileged Texas Hold'em identity for pairing validation only."""

    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    if task is None:
        raise RuntimeError("VLABench endpoint did not expose a live Texas Hold'em task")
    pokers = tuple(getattr(task, "pokers", ()))
    if not pokers:
        raise RuntimeError("Texas Hold'em task did not expose task.pokers")
    card_names = tuple(str(getattr(poker, "name", "")) for poker in pokers)
    raw_targets = getattr(task, "target_entities", None)
    if isinstance(raw_targets, Mapping) or (
        isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes))
    ):
        target_names = tuple(str(name) for name in raw_targets)
    else:
        raise TypeError("Texas Hold'em task did not expose target_entities as names")
    return TexasHoldemDealIdentity(
        card_names=card_names,
        target_card_names=target_names,
        hand_type=str(getattr(task, "max_cardtype", "")),
    )


def _step_main_endpoint(
    endpoint: Any,
    waypoint: PrivilegedSkillWaypoint,
    *,
    phase: str,
    repeat_index: int | None,
    evaluation_label: str,
    is_oracle_upper_bound: bool,
    selected_card_names: tuple[str, ...],
    action_clock: Callable[[], float],
) -> EpisodeStep:
    action = RobotAction(
        timestamp_s=float(action_clock()),
        values={"action": waypoint.action},
        metadata={
            "controller": SIMULATOR_PRIVILEGED_SKILL_EXECUTOR,
            "evaluation_label": evaluation_label,
            "oracle_upper_bound": is_oracle_upper_bound,
            "privileged_simulator_state": True,
            "deployable": False,
            "selected_card_names": selected_card_names,
            "phase": phase,
            "skill_index": waypoint.skill_index,
            "waypoint_index": waypoint.waypoint_index,
            "repeat_index": repeat_index,
        },
    )
    outcome = endpoint.step(action)
    if not isinstance(outcome, EpisodeStep):
        raise TypeError("main simulator endpoint step() must return EpisodeStep")
    return outcome


def _get_selected_expert_skill_sequence(
    task: Any,
    physics: Any,
    selected_card_names: tuple[str, ...],
) -> Any:
    """Ask the official task for skills after overriding only shadow targets."""

    get_sequence = getattr(task, "get_expert_skill_sequence", None)
    if not callable(get_sequence):
        raise TypeError("Texas Hold'em task did not expose get_expert_skill_sequence()")
    entities = getattr(task, "entities", None)
    if not isinstance(entities, Mapping):
        raise TypeError("Texas Hold'em task did not expose task.entities as a mapping")
    original_targets = getattr(task, "_target_entities", None)
    if not isinstance(original_targets, Mapping):
        raise TypeError("Texas Hold'em task did not expose mutable shadow _target_entities")
    missing = set(selected_card_names).difference(entities)
    if missing:
        raise RuntimeError(
            f"selected cards are missing from shadow task.entities: {sorted(missing)}"
        )

    selected_targets = {name: entities[name] for name in selected_card_names}
    task._target_entities = selected_targets
    try:
        return get_sequence(physics)
    finally:
        task._target_entities = original_targets


def _robot_base_xyz(raw_environment: Any, inner_environment: Any) -> Any:
    base = getattr(raw_environment, "_robot_base_xyz", None)
    if base is not None:
        return base
    get_position = getattr(inner_environment, "get_robot_frame_position", None)
    if not callable(get_position):
        raise TypeError("shadow VLABench environment did not expose robot base position")
    return get_position()


def _endpoint_fingerprint(endpoint: Any) -> str:
    fingerprint = getattr(endpoint, "initial_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise RuntimeError("VLABench endpoint must be reset before oracle pairing")
    return fingerprint.strip()


def _validate_endpoint_seed_if_available(endpoint: Any, expected_seed: int) -> None:
    observe = getattr(endpoint, "observe", None)
    if not callable(observe):
        return
    observation = observe()
    metadata = getattr(observation, "metadata", None)
    if not isinstance(metadata, Mapping):
        return
    actual_seed = metadata.get("seed")
    if actual_seed is not None and actual_seed != expected_seed:
        raise RuntimeError(
            f"VLABench endpoint was reset with seed {actual_seed}, expected {expected_seed}"
        )


def _validate_selected_cards(
    selected_cards: tuple[str, ...],
    deal: TexasHoldemDealIdentity,
) -> None:
    missing = set(selected_cards).difference(deal.card_names)
    if missing:
        raise RuntimeError(
            f"selected cards are missing from the paired poker deal: {sorted(missing)}"
        )


def _non_empty_unique_names(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    names = tuple(values)
    if not names:
        raise ValueError(f"{field_name} must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{field_name} must contain non-empty strings")
    normalized = tuple(name.strip() for name in names)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _finite_non_negative(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return normalized


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "VLABench privileged oracle execution requires NumPy in the simulator environment"
        ) from error
    return np


# Backward names remain import-compatible, but the canonical types deliberately
# use "Skill" rather than "Oracle": plan selection and privileged execution are
# separate experimental variables.
PrivilegedOracleWaypoint = PrivilegedSkillWaypoint
PrivilegedOracleTrajectory = PrivilegedSkillTrajectory
PrivilegedOracleReplay = PrivilegedSkillReplay
