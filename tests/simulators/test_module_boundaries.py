"""Public import identity checks for focused simulator modules."""

from embodied_runtime import simulators
from embodied_runtime.simulators import vlabench_privileged_oracle
from embodied_runtime.simulators.gym_adapter import GymLikeSimulatorAdapter
from embodied_runtime.simulators.trace_recorder import TraceRecorder
from embodied_runtime.simulators.trace_values import ResetTraceEvent
from embodied_runtime.simulators.vlabench_fingerprint import observation_fingerprint
from embodied_runtime.simulators.vlabench_privileged_generation import (
    convert_vlabench_expert_waypoint,
    generate_texas_holdem_privileged_skill_trajectory,
)
from embodied_runtime.simulators.vlabench_privileged_inspection import (
    inspect_texas_holdem_deal,
)
from embodied_runtime.simulators.vlabench_privileged_replay import (
    replay_privileged_skill_trajectory,
)
from embodied_runtime.simulators.vlabench_privileged_values import (
    PrivilegedSkillTrajectory,
)


def test_package_exports_are_canonical_module_values() -> None:
    assert simulators.GymLikeSimulatorAdapter is GymLikeSimulatorAdapter
    assert simulators.TraceRecorder is TraceRecorder
    assert simulators.ResetTraceEvent is ResetTraceEvent
    assert simulators.observation_fingerprint is observation_fingerprint
    assert simulators.PrivilegedSkillTrajectory is PrivilegedSkillTrajectory


def test_established_oracle_module_imports_keep_their_identity() -> None:
    assert (
        vlabench_privileged_oracle.convert_vlabench_expert_waypoint
        is convert_vlabench_expert_waypoint
    )
    assert (
        vlabench_privileged_oracle.generate_texas_holdem_privileged_skill_trajectory
        is generate_texas_holdem_privileged_skill_trajectory
    )
    assert vlabench_privileged_oracle.inspect_texas_holdem_deal is inspect_texas_holdem_deal
    assert (
        vlabench_privileged_oracle.replay_privileged_skill_trajectory
        is replay_privileged_skill_trajectory
    )
