"""Model-agnostic hardware interfaces, values, and implementations."""

from .artifact import ArtifactVariant
from .ascend import AscendBackend
from .compile import CompileOptions
from .device import DeviceInfo
from .errors import BackendExecutionError, UnsupportedBackendError
from .horizon import HorizonBackend
from .interfaces import Backend, BackendSession
from .memory import MemoryStats
from .registry import BackendRegistry
from .requirements import ResourceRequirements
from .support import SupportReport
from .torch_cuda import TorchBackendSession, TorchCudaBackend

__all__ = [
    "ArtifactVariant",
    "AscendBackend",
    "Backend",
    "BackendExecutionError",
    "BackendRegistry",
    "BackendSession",
    "CompileOptions",
    "DeviceInfo",
    "HorizonBackend",
    "MemoryStats",
    "ResourceRequirements",
    "SupportReport",
    "TorchBackendSession",
    "TorchCudaBackend",
    "UnsupportedBackendError",
]
