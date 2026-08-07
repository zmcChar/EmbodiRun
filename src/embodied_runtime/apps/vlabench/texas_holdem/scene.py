"""Texas Hold'em simulator scene inspection and coordinate conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from embodied_runtime.robots.observation import RobotObservation

from .settings import EndpointPlaceControllerConfig, PokerCard, PokerDeal

TARGET_CONTAINER_NAME = "target_container"


def inspect_poker_deal(endpoint: Any) -> PokerDeal:
    task, _ = task_and_physics(endpoint)
    pokers = tuple(getattr(task, "pokers", ()))
    if not pokers:
        raise RuntimeError("texas_holdem task did not expose task.pokers")
    cards = tuple(
        PokerCard(
            name=str(getattr(poker, "name", "")),
            value=str(getattr(poker, "value", "")),
            suit=str(getattr(poker, "suite", "")),
        )
        for poker in pokers
    )
    raw_targets = getattr(task, "target_entities", None)
    if isinstance(raw_targets, Mapping) or (
        isinstance(raw_targets, Sequence) and not isinstance(raw_targets, (str, bytes))
    ):
        target_names = tuple(str(name) for name in raw_targets)
    else:
        raise TypeError("texas_holdem task did not expose target_entities as a sequence")
    return PokerDeal(
        cards=cards,
        target_names=target_names,
        hand_type=str(getattr(task, "max_cardtype", "")),
    )


def grasped_poker_name(endpoint: Any) -> str | None:
    task, physics = task_and_physics(endpoint)
    robot = getattr(task, "robot", None)
    if robot is None:
        raise RuntimeError("texas_holdem task did not expose its robot")
    for poker in tuple(getattr(task, "pokers", ())):
        is_grasped = getattr(poker, "is_grasped", None)
        if callable(is_grasped) and bool(is_grasped(physics, robot)):
            return str(poker.name)
    return None


def contained_target_names(endpoint: Any, deal: PokerDeal) -> frozenset[str]:
    return contained_card_names(endpoint, deal.target_names)


def contained_card_names(endpoint: Any, card_names: Sequence[str]) -> frozenset[str]:
    task, physics = task_and_physics(endpoint)
    entities = getattr(task, "entities", None)
    if not isinstance(entities, Mapping):
        raise TypeError("texas_holdem task did not expose task.entities as a mapping")
    container = entities.get(TARGET_CONTAINER_NAME)
    contain = getattr(container, "contain", None)
    if not callable(contain):
        raise TypeError("Texas Hold'em placemat did not expose contain()")

    completed: set[str] = set()
    for name in card_names:
        entity = entities.get(name)
        get_xpos = getattr(entity, "get_xpos", None)
        if not callable(get_xpos):
            raise TypeError(f"Texas Hold'em target {name!r} did not expose get_xpos()")
        if bool(contain(get_xpos(physics), physics)):
            completed.add(name)
    return frozenset(completed)


def task_and_physics(endpoint: Any) -> tuple[Any, Any]:
    environment = endpoint.raw_environment
    inner = getattr(environment, "_env", None)
    task = getattr(inner, "task", None)
    physics = getattr(inner, "physics", None)
    if task is None or physics is None:
        raise RuntimeError("VLABench endpoint did not expose a live task and physics")
    return task, physics


def place_target_robot_frame(
    endpoint: Any,
    *,
    placement_index: int,
    config: EndpointPlaceControllerConfig,
) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error

    environment = endpoint.raw_environment
    task, physics = task_and_physics(endpoint)
    entities = getattr(task, "entities", None)
    if not isinstance(entities, Mapping):
        raise TypeError("texas_holdem task did not expose task.entities as a mapping")
    container = entities.get(TARGET_CONTAINER_NAME)
    get_place_point = getattr(container, "get_place_point", None)
    if not callable(get_place_point):
        raise TypeError("Texas Hold'em placemat did not expose get_place_point()")
    place_points = list(get_place_point(physics))
    if not place_points:
        raise RuntimeError("Texas Hold'em placemat did not return a place point")
    target_world = np.asarray(place_points[-1], dtype=np.float64).reshape(3).copy()
    slot = placement_index % config.slot_count
    target_world[1] += (slot - (config.slot_count - 1) / 2.0) * config.slot_spacing_m

    robot = getattr(task, "robot", None)
    ee_offset = getattr(robot, "ee_offset", None)
    if not callable(ee_offset):
        raise TypeError("VLABench robot did not expose ee_offset()")
    target_world += np.asarray(ee_offset(physics), dtype=np.float64).reshape(3)

    base = getattr(environment, "_robot_base_xyz", None)
    if base is None:
        get_robot_frame_position = getattr(
            getattr(environment, "_env", None), "get_robot_frame_position", None
        )
        if not callable(get_robot_frame_position):
            raise RuntimeError("VLABench environment did not expose its robot base position")
        base = get_robot_frame_position()
    target_robot = target_world - np.asarray(base, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(target_robot)):
        raise RuntimeError("place target contains non-finite coordinates")
    if np.max(np.abs(target_robot)) > config.workspace_abs_limit_m:
        raise RuntimeError(
            f"place target {target_robot.tolist()} exceeds configured workspace limit"
        )
    return target_robot


def agent_state(observation: RobotObservation) -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("the VLABench place controller requires NumPy") from error
    if not isinstance(observation.values, Mapping) or "agent_pos" not in observation.values:
        raise RuntimeError("VLABench observation did not contain agent_pos")
    state = np.asarray(observation.values["agent_pos"], dtype=np.float64).reshape(-1)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise RuntimeError("VLABench agent_pos must be a finite 7-D vector")
    return state
