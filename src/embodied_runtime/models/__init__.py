"""Model-family adapters and their lazy registry.

This package depends only on :mod:`embodied_runtime.contracts`.  In particular, model
adapters must not import the execution engine or a hardware backend.
"""

from .base import BaseModelAdapter
from .registry import available_models, get_model_adapter, register_model

__all__ = [
    "BaseModelAdapter",
    "available_models",
    "get_model_adapter",
    "register_model",
]
