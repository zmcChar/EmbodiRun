"""Canonical model-domain values, plans, adapters, and lazy registry.

Model adapters own model semantics and portable entrypoints. They do not import
the execution engine or concrete hardware backends.
"""

from .action import ActionChunk
from .base import BaseModelAdapter
from .errors import ModelPackageError
from .interfaces import ModelAdapter
from .package import ModelPackage
from .plans import ExecutionPlan, IterativeFlowPlan, SingleForwardPlan
from .registry import available_models, get_model_adapter, register_model
from .request import RawRequest
from .spec import EntrypointSpec, ModelSpec

__all__ = [
    "ActionChunk",
    "BaseModelAdapter",
    "EntrypointSpec",
    "ExecutionPlan",
    "IterativeFlowPlan",
    "ModelAdapter",
    "ModelPackage",
    "ModelPackageError",
    "ModelSpec",
    "RawRequest",
    "SingleForwardPlan",
    "available_models",
    "get_model_adapter",
    "register_model",
]
