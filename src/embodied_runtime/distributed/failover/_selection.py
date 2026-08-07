"""Pure result-authority selection for asynchronous failover."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import (
    FailoverConfig,
    FailoverDecision,
    FailoverMode,
    FallbackReason,
    ResultFuser,
    ResultSource,
)

ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class CloudResult(Generic[ResultT]):
    result: ResultT
    sequence_id: int
    connection_epoch: int
    submitted_at_s: float


@dataclass(frozen=True, slots=True)
class SelectionOutcome(Generic[ResultT]):
    decision: FailoverDecision[ResultT]
    discard_cloud_result: bool = False


def select_available_cloud_result(
    *,
    edge_result: ResultT,
    sequence_id: int,
    epoch: int,
    cloud_result: CloudResult[ResultT] | None,
    last_cloud_failure: tuple[int, FallbackReason] | None,
    cloud_request_in_flight: bool,
    config: FailoverConfig,
    fuser: ResultFuser[ResultT] | None,
    clock: Callable[[], float],
) -> SelectionOutcome[ResultT]:
    if cloud_result is None:
        reason = (
            last_cloud_failure[1]
            if last_cloud_failure is not None and last_cloud_failure[0] == epoch
            else FallbackReason.CLOUD_PENDING
            if cloud_request_in_flight
            else FallbackReason.NO_CLOUD_RESULT
        )
        return SelectionOutcome(edge_decision(edge_result, sequence_id, epoch, reason))

    age_s = clock() - cloud_result.submitted_at_s
    sequence_lag = sequence_id - cloud_result.sequence_id
    if (
        cloud_result.connection_epoch != epoch
        or age_s > config.cloud_result_ttl_s
        or sequence_lag > config.max_cloud_sequence_lag
    ):
        return SelectionOutcome(
            edge_decision(
                edge_result,
                sequence_id,
                epoch,
                FallbackReason.STALE_CLOUD_RESULT,
            ),
            discard_cloud_result=True,
        )

    if config.mode is FailoverMode.ASYNC_BLEND:
        assert fuser is not None
        try:
            selected_result = fuser(edge_result, cloud_result.result)
            if inspect.isawaitable(selected_result):
                close = getattr(selected_result, "close", None)
                if callable(close):
                    close()
                raise TypeError("result fuser must be synchronous")
        except Exception:  # noqa: BLE001 - a user fuser cannot stop edge control
            return SelectionOutcome(
                edge_decision(
                    edge_result,
                    sequence_id,
                    epoch,
                    FallbackReason.BLEND_ERROR,
                ),
                discard_cloud_result=True,
            )
        selected_source = ResultSource.BLENDED
    else:
        selected_result = cloud_result.result
        selected_source = ResultSource.CLOUD

    return SelectionOutcome(
        FailoverDecision(
            result=selected_result,
            source=selected_source,
            sequence_id=sequence_id,
            source_sequence_id=cloud_result.sequence_id,
            connection_epoch=epoch,
            cloud_result_age_s=max(0.0, age_s),
        )
    )


def edge_decision(
    result: ResultT,
    sequence_id: int,
    epoch: int,
    reason: FallbackReason,
) -> FailoverDecision[ResultT]:
    return FailoverDecision(
        result=result,
        source=ResultSource.EDGE,
        sequence_id=sequence_id,
        source_sequence_id=sequence_id,
        connection_epoch=epoch,
        fallback_reason=reason,
    )
