"""Stable contracts shared by independently owned domains."""

from .artifact import ArtifactVariant, CompileOptions
from .backend import Backend, BackendSession, SupportReport
from .errors import (
    BackendExecutionError,
    ModelPackageError,
    RequestCancelledError,
    RequestDeadlineExceededError,
    RuntimePrototypeError,
    UnsupportedBackendError,
)
from .execution import ExecutionContext, InferenceRequest, InferenceResult, RequestStatus
from .model import (
    ActionChunk,
    EntrypointSpec,
    ModelAdapter,
    ModelPackage,
    ModelSpec,
    RawRequest,
)
from .plan import ExecutionPlan, FlowRecipe, IterativeFlowPlan, SingleForwardPlan
from .resource import DeviceInfo, MemoryStats, ResourceRequirements
from .robot import RobotAction, RobotAdapter, RobotObservation
from .task import (
    PlanEnvelope,
    PlanFeedback,
    PlanRequest,
    PlanStep,
    PlanStepStatus,
    TaskGoal,
)
from .transport import Envelope
from .types import Metadata, Shape, TensorTree

__all__ = [
    "ActionChunk",
    "ArtifactVariant",
    "Backend",
    "BackendExecutionError",
    "BackendSession",
    "CompileOptions",
    "DeviceInfo",
    "EntrypointSpec",
    "Envelope",
    "ExecutionContext",
    "ExecutionPlan",
    "FlowRecipe",
    "InferenceRequest",
    "InferenceResult",
    "IterativeFlowPlan",
    "MemoryStats",
    "Metadata",
    "ModelAdapter",
    "ModelPackage",
    "ModelPackageError",
    "ModelSpec",
    "PlanEnvelope",
    "PlanFeedback",
    "PlanRequest",
    "PlanStep",
    "PlanStepStatus",
    "RawRequest",
    "RequestCancelledError",
    "RequestDeadlineExceededError",
    "RequestStatus",
    "ResourceRequirements",
    "RobotAction",
    "RobotAdapter",
    "RobotObservation",
    "RuntimePrototypeError",
    "Shape",
    "SingleForwardPlan",
    "SupportReport",
    "TaskGoal",
    "TensorTree",
    "UnsupportedBackendError",
]
