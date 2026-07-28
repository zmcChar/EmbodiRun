"""Runner for a single backend entrypoint."""

from __future__ import annotations

from collections.abc import Sequence

from embodied_runtime.contracts import ExecutionPlan, SingleForwardPlan, TensorTree

from ..errors import EnginePayloadError
from ..plan_runner import BatchAborted, RunnerHost, RunnerRequest


class SingleForwardRunner:
    """Delegate a one-stage plan through the engine's backend submission path."""

    def __init__(self, plan: ExecutionPlan, host: RunnerHost) -> None:
        if not isinstance(plan, SingleForwardPlan):
            raise TypeError("SingleForwardRunner requires SingleForwardPlan")
        self.plan = plan
        self.host = host

    def run_sync(self, request: RunnerRequest) -> TensorTree:
        self._reject_num_steps([request])
        self.host.ensure_active_sync(request)
        output = self.host.submit_sync(
            self.plan.forward,
            self.host.preprocessed_payload(request),
            request,
            None,
            None,
        )
        self.host.ensure_active_sync(request)
        return output

    async def run_async(
        self,
        requests: Sequence[RunnerRequest],
        payload: TensorTree,
    ) -> TensorTree:
        self._reject_num_steps(requests)
        if not self.host.has_active(requests):
            raise BatchAborted
        output = await self.host.submit_async(
            self.plan.forward,
            payload,
            requests,
            None,
            None,
        )
        if not self.host.has_active(requests):
            raise BatchAborted
        return output

    @staticmethod
    def _reject_num_steps(requests: Sequence[RunnerRequest]) -> None:
        if any(request.request.num_steps is not None for request in requests):
            raise EnginePayloadError("num_steps is not valid for a single-forward execution plan")
