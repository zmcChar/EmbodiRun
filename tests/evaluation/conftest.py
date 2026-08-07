from __future__ import annotations

from collections.abc import Callable

import pytest

from embodied_runtime.evaluation.reporting import ExperimentCondition, TrialRecord


@pytest.fixture
def trial_factory() -> Callable[..., TrialRecord]:
    def make_trial(
        pair: int,
        condition: ExperimentCondition,
        success: bool,
        *,
        edge: tuple[float, ...] = (1.0,),
        planner: tuple[float, ...] = (),
        chunks: tuple[float, ...] = (),
        queue_hits: tuple[float, ...] = (),
        deadline_misses: int = 0,
    ) -> TrialRecord:
        return TrialRecord(
            pair_id=f"pair-{pair:02d}",
            task_id="composite-pick",
            seed=pair,
            condition=condition,
            success=success,
            steps=10 + pair,
            edge_latencies_ms=edge,
            planner_latencies_ms=planner,
            chunk_generation_latencies_ms=chunks,
            queue_hit_latencies_ms=queue_hits,
            deadline_misses=deadline_misses,
        )

    return make_trial
