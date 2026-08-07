"""Failover policy values and immutable decisions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from embodied_runtime._compat import StrEnum

ResultT = TypeVar("ResultT")
ResultFuser = Callable[[ResultT, ResultT], ResultT]


class FailoverMode(StrEnum):
    """Supported result-selection policies."""

    ASYNC_CLOUD_PREFERRED = "async_cloud_preferred"
    ASYNC_BLEND = "async_blend"
    EDGE_ONLY = "edge_only"


class ResultSource(StrEnum):
    CLOUD = "cloud"
    EDGE = "edge"
    BLENDED = "blended"


class FallbackReason(StrEnum):
    CLOUD_DISABLED = "cloud_disabled"
    CLOUD_DISCONNECTED = "cloud_disconnected"
    CLOUD_PENDING = "cloud_pending"
    CLOUD_TIMEOUT = "cloud_timeout"
    CLOUD_ERROR = "cloud_error"
    BLEND_ERROR = "blend_error"
    NO_CLOUD_RESULT = "no_cloud_result"
    STALE_CLOUD_RESULT = "stale_cloud_result"


@dataclass(frozen=True, slots=True)
class FailoverConfig:
    """Policy knobs for non-blocking cloud-preferred inference.

    A control tick never waits for cloud completion. ``cloud_result_ttl_s`` and
    ``max_cloud_sequence_lag`` bound reuse of the most recent asynchronous cloud
    result. At most one cloud request is active, so a slow or disconnected cloud
    cannot create an unbounded request backlog.
    """

    mode: FailoverMode = FailoverMode.ASYNC_CLOUD_PREFERRED
    cloud_request_timeout_s: float = 5.0
    cloud_result_ttl_s: float = 1.0
    max_cloud_sequence_lag: int = 1
    cloud_submit_interval_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.cloud_request_timeout_s) or self.cloud_request_timeout_s <= 0:
            raise ValueError("cloud_request_timeout_s must be finite and greater than zero")
        if not math.isfinite(self.cloud_result_ttl_s) or self.cloud_result_ttl_s <= 0:
            raise ValueError("cloud_result_ttl_s must be finite and greater than zero")
        if self.max_cloud_sequence_lag < 0:
            raise ValueError("max_cloud_sequence_lag cannot be negative")
        if not math.isfinite(self.cloud_submit_interval_s) or self.cloud_submit_interval_s < 0:
            raise ValueError("cloud_submit_interval_s must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FailoverDecision(Generic[ResultT]):
    """One control-tick result plus the reason for its authority source."""

    result: ResultT
    source: ResultSource
    sequence_id: int
    source_sequence_id: int
    connection_epoch: int
    fallback_reason: FallbackReason | None = None
    cloud_result_age_s: float | None = None
