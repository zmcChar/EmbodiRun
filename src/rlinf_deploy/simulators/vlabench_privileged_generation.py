"""Generate privileged VLABench skills inside a paired shadow environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .vlabench import VLABenchSimulatorEndpoint
from .vlabench_privileged_inspection import (
    endpoint_fingerprint,
    inspect_texas_holdem_deal,
    validate_endpoint_seed_if_available,
    validate_selected_cards,
)
from .vlabench_privileged_values import (
    DEFAULT_FINGER_TOLERANCE,
    DEFAULT_GRIPPER_OPEN_THRESHOLD,
    TEXAS_HOLDEM_TASK,
    PrivilegedSkillTrajectory,
    PrivilegedSkillWaypoint,
    finite_non_negative,
    non_empty_unique_names,
    validate_seed,
)


def convert_vlabench_expert_waypoint(
    waypoint: Any,
    *,
    robot_base_xyz: Any,
    gripper_open_threshold: float = DEFAULT_GRIPPER_OPEN_THRESHOLD,
    finger_tolerance: float = DEFAULT_FINGER_TOLERANCE,
) -> tuple[float, float, float, float, float, float, float]:
    """Convert VLABench expert ``xyz_world+euler+two_fingers`` to LeRobot 7D."""

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
    threshold = finite_non_negative(
        gripper_open_threshold,
        field_name="gripper_open_threshold",
    )
    tolerance = finite_non_negative(finger_tolerance, field_name="finger_tolerance")
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
    """Generate selected-card expert waypoints in an identically seeded shadow env."""

    validate_seed(seed)
    selected_cards = non_empty_unique_names(
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

    main_fingerprint = endpoint_fingerprint(main_endpoint)
    main_deal = inspect_texas_holdem_deal(main_endpoint)
    validate_selected_cards(selected_cards, main_deal)
    validate_endpoint_seed_if_available(main_endpoint, seed)

    factory = shadow_endpoint_factory or VLABenchSimulatorEndpoint
    shadow_endpoint = factory(
        TEXAS_HOLDEM_TASK,
        max_episode_steps=shadow_max_episode_steps,
    )
    try:
        shadow_endpoint.reset(seed=seed)
        shadow_fingerprint = endpoint_fingerprint(shadow_endpoint)
        shadow_deal = inspect_texas_holdem_deal(shadow_endpoint)
        validate_endpoint_seed_if_available(shadow_endpoint, seed)
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


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "VLABench privileged oracle execution requires NumPy in the simulator environment"
        ) from error
    return np


__all__ = [
    "convert_vlabench_expert_waypoint",
    "generate_texas_holdem_privileged_skill_trajectory",
]
