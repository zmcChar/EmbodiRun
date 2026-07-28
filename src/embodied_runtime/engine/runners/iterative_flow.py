"""Runner for staged flow-matching plans."""

from __future__ import annotations

from collections.abc import Sequence

from embodied_runtime.contracts import ExecutionPlan, IterativeFlowPlan, TensorTree

from ..plan_runner import BatchAborted, RunnerHost, RunnerRequest


class IterativeFlowRunner:
    """Execute encode, initialization, iterative integration, and finalize."""

    def __init__(self, plan: ExecutionPlan, host: RunnerHost) -> None:
        if not isinstance(plan, IterativeFlowPlan):
            raise TypeError("IterativeFlowRunner requires IterativeFlowPlan")
        self.plan = plan
        self.host = host

    def run_sync(self, request: RunnerRequest) -> TensorTree:
        payload = self.host.preprocessed_payload(request)
        prefix = self.host.submit_sync(
            self.plan.encode,
            payload,
            request,
            None,
            None,
        )
        state = self.host.submit_sync(
            self.plan.initialize,
            {"batch_size": 1, "seed": request.request.seed},
            request,
            None,
            None,
        )
        state = _extract_named(state, "state")

        schedule = self.plan.schedule(request.request.num_steps)
        for index, (time_value, dt) in enumerate(schedule):
            self.host.ensure_active_sync(request)
            step_output = self.host.submit_sync(
                self.plan.step,
                {"state": state, "time": time_value, "prefix": prefix},
                request,
                index,
                len(schedule),
            )
            if self.plan.step_returns_state:
                state = _extract_named(step_output, "state")
            else:
                state = self.host.advance_sync(
                    state,
                    _extract_named(step_output, "velocity"),
                    dt,
                    request,
                    index,
                    len(schedule),
                )
            if self.plan.safe_point_after_step:
                self.host.ensure_active_sync(request)
                self.host.check_memory_safe_point_sync()

        self.host.ensure_active_sync(request)
        return self.host.submit_sync(
            self.plan.finalize,
            {"state": state},
            request,
            None,
            len(schedule),
        )

    async def run_async(
        self,
        requests: Sequence[RunnerRequest],
        payload: TensorTree,
    ) -> TensorTree:
        if not self.host.has_active(requests):
            raise BatchAborted

        prefix = await self.host.submit_async(
            self.plan.encode,
            payload,
            requests,
            None,
            None,
        )
        state = await self.host.submit_async(
            self.plan.initialize,
            # Explicitly seeded requests are isolated from dynamic batches, so
            # the entrypoint receives the same scalar seed shape in both modes.
            {"batch_size": len(requests), "seed": requests[0].request.seed},
            requests,
            None,
            None,
        )
        state = _extract_named(state, "state")

        schedule = self.plan.schedule(requests[0].request.num_steps)
        for index, (time_value, dt) in enumerate(schedule):
            if not self.host.has_active(requests):
                raise BatchAborted
            step_output = await self.host.submit_async(
                self.plan.step,
                {"state": state, "time": time_value, "prefix": prefix},
                requests,
                index,
                len(schedule),
            )
            if self.plan.step_returns_state:
                state = _extract_named(step_output, "state")
            else:
                state = await self.host.advance_async(
                    state,
                    _extract_named(step_output, "velocity"),
                    dt,
                    requests,
                    index,
                    len(schedule),
                )
            if self.plan.safe_point_after_step:
                if not self.host.has_active(requests):
                    raise BatchAborted
                await self.host.check_memory_safe_point_async()

        if not self.host.has_active(requests):
            raise BatchAborted
        return await self.host.submit_async(
            self.plan.finalize,
            {"state": state},
            requests,
            None,
            len(schedule),
        )


def _extract_named(value: TensorTree, name: str) -> TensorTree:
    if isinstance(value, dict) and name in value:
        return value[name]
    return value
