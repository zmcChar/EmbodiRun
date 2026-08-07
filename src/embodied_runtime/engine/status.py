"""Inference request lifecycle states."""

from embodied_runtime._compat import StrEnum


class RequestStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


__all__ = ["RequestStatus"]
