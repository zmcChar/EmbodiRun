"""Model-agnostic hardware backends (Group 4).

Concrete backends depend only on :mod:`embodied_runtime.contracts`.  Model packages
and the execution engine meet here through the stable ``Backend`` and
``BackendSession`` protocols; no model-family dispatch belongs in this package.
"""

from .ascend import AscendBackend
from .horizon import HorizonBackend
from .registry import BackendRegistry
from .torch_cuda import TorchCudaBackend, TorchBackendSession

__all__ = [
    "AscendBackend",
    "BackendRegistry",
    "HorizonBackend",
    "TorchBackendSession",
    "TorchCudaBackend",
]
