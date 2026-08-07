"""Generic PyTorch backend with eager, compile, and selected CUDA Graph execution."""

from __future__ import annotations

import importlib
import platform
from typing import Any

from embodied_runtime.models.package import ModelPackage

from ..artifact import ArtifactVariant
from ..compile import CompileOptions
from ..device import DeviceInfo
from ..errors import UnsupportedBackendError
from ..support import SupportReport
from .compiler import (
    TorchArtifactPayload,
    make_payload,
    move_runtime_module,
    normalize_mode,
    resolve_dtype,
)
from .lifecycle import acquire_runtime_module
from .memory import _cpu_memory
from .session import TorchBackendSession


def _load_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


class TorchCudaBackend:
    """Run any callable ``ModelPackage`` on PyTorch CPU or NVIDIA CUDA.

    CPU support exists both as a useful reference path and as a way to validate
    the contract without a GPU.  It does not imply CPU performance suitability
    for a full VLA.
    """

    name = "torch_cuda"

    def __init__(self) -> None:
        self._torch = _load_torch()

    def probe(self) -> tuple[DeviceInfo, ...]:
        torch = self._torch
        if torch is None:
            return ()

        total_cpu, _ = _cpu_memory()
        cpu_capabilities = {"torch", "eager"}
        if hasattr(torch, "compile"):
            cpu_capabilities.add("torch.compile")
        devices = [
            DeviceInfo(
                backend=self.name,
                device_id="cpu",
                kind="cpu",
                vendor=platform.processor() or platform.machine() or "generic",
                name=platform.machine() or "CPU",
                total_memory_bytes=total_cpu,
                capabilities=frozenset(cpu_capabilities),
                metadata={"torch_version": str(torch.__version__)},
            )
        ]

        try:
            cuda_available = torch.cuda.is_available()
            cuda_count = torch.cuda.device_count() if cuda_available else 0
        except (RuntimeError, AssertionError):
            cuda_count = 0
        for index in range(cuda_count):
            try:
                properties = torch.cuda.get_device_properties(index)
                major, minor = torch.cuda.get_device_capability(index)
                capabilities = {
                    "torch",
                    "cuda",
                    "eager",
                    f"compute_{major}{minor}",
                }
                if hasattr(torch, "compile"):
                    capabilities.add("torch.compile")
                if hasattr(torch.cuda, "CUDAGraph") and hasattr(torch.cuda, "graph"):
                    capabilities.add("cuda_graph")
                try:
                    with torch.cuda.device(index):
                        bf16_supported = torch.cuda.is_bf16_supported(including_emulation=False)
                except TypeError:
                    with torch.cuda.device(index):
                        bf16_supported = torch.cuda.is_bf16_supported()
                if bf16_supported:
                    capabilities.add("bf16")
                devices.append(
                    DeviceInfo(
                        backend=self.name,
                        device_id=f"cuda:{index}",
                        kind="accelerator",
                        vendor="NVIDIA",
                        name=properties.name,
                        total_memory_bytes=int(properties.total_memory),
                        capabilities=frozenset(capabilities),
                        metadata={
                            "torch_version": str(torch.__version__),
                            "cuda_version": str(torch.version.cuda),
                            "compute_capability": f"{major}.{minor}",
                        },
                    )
                )
            except (RuntimeError, AssertionError):
                continue
        return tuple(devices)

    def _device_is_available(self, device: DeviceInfo) -> tuple[bool, str | None]:
        torch = self._torch
        if torch is None:
            return False, "PyTorch is not installed"
        if device.backend != self.name:
            return False, (f"device belongs to backend {device.backend!r}, expected {self.name!r}")
        if device.device_id == "cpu":
            return True, None
        if not device.device_id.startswith("cuda:"):
            return False, f"unsupported PyTorch device id {device.device_id!r}"
        try:
            index = int(device.device_id.partition(":")[2])
        except ValueError:
            return False, f"invalid CUDA device id {device.device_id!r}"
        try:
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except (RuntimeError, AssertionError) as error:
            return False, f"CUDA probe failed: {error}"
        if not 0 <= index < count:
            return False, f"CUDA device index {index} is unavailable (device_count={count})"
        return True, None

    def supports(self, package: ModelPackage, device: DeviceInfo) -> SupportReport:
        available, reason = self._device_is_available(device)
        if not available:
            return SupportReport.no(reason or "device unavailable")
        non_callable = sorted(
            name for name, entrypoint in package.entrypoints.items() if not callable(entrypoint)
        )
        if non_callable:
            return SupportReport.no(
                f"package entrypoints are not callable: {', '.join(non_callable)}"
            )
        if not package.entrypoints:
            return SupportReport.no("package has no executable entrypoints")
        return SupportReport.yes(*sorted(device.capabilities))

    def compile(
        self,
        package: ModelPackage,
        device: DeviceInfo,
        options: CompileOptions,
    ) -> ArtifactVariant:
        report = self.supports(package, device)
        if not report.supported:
            raise UnsupportedBackendError("; ".join(report.reasons))
        torch = self._torch
        if torch is None:  # guarded by supports, retained for type narrowing
            raise UnsupportedBackendError("PyTorch is not installed")

        normalize_mode(options.mode)
        dtype = resolve_dtype(torch, options.dtype)
        if device.device_id.startswith("cuda") and dtype is torch.bfloat16:
            try:
                with torch.cuda.device(torch.device(device.device_id)):
                    bf16_supported = torch.cuda.is_bf16_supported()
            except (RuntimeError, AssertionError):
                bf16_supported = False
            if not bf16_supported:
                raise UnsupportedBackendError(
                    f"{device.device_id} does not report bfloat16 support"
                )

        payload = make_payload(
            torch,
            package,
            options,
            device_id=device.device_id,
        )
        metadata = {
            "package_id": payload.package_id,
            "requested_mode": payload.requested_mode,
            "actual_mode": payload.actual_mode,
            "dtype": None if dtype is None else str(dtype),
            "compile_failures": dict(payload.compile_failures),
            "cuda_graph_entrypoints": tuple(sorted(payload.cuda_graph.entrypoints)),
        }
        return ArtifactVariant(
            backend=self.name,
            device=device,
            package=package,
            options=options,
            payload=payload,
            metadata=metadata,
        )

    def load(self, artifact: ArtifactVariant) -> TorchBackendSession:
        if artifact.backend != self.name:
            raise UnsupportedBackendError(
                f"cannot load {artifact.backend!r} artifact with {self.name!r}"
            )
        if artifact.device.backend != self.name:
            raise UnsupportedBackendError(f"artifact device belongs to {artifact.device.backend!r}")
        if not isinstance(artifact.payload, TorchArtifactPayload):
            raise UnsupportedBackendError("artifact payload was not produced by TorchCudaBackend")
        torch = self._torch
        if torch is None:
            raise UnsupportedBackendError("PyTorch is not installed")
        available, reason = self._device_is_available(artifact.device)
        if not available:
            raise UnsupportedBackendError(reason or "artifact device is unavailable")

        payload = artifact.payload.for_session()
        dtype_policy = "preserve" if payload.dtype is None else str(payload.dtype)
        module_lease = acquire_runtime_module(
            payload.runtime_module,
            dtype_policy=dtype_policy,
        )
        try:
            move_runtime_module(
                torch,
                payload.runtime_module,
                device_id=payload.device_id,
                dtype=payload.dtype,
            )
            session = TorchBackendSession(
                torch,
                artifact.device,
                payload,
                release_module=module_lease.release,
            )
            module_lease.commit_dtype_policy()
            return session
        except Exception:
            module_lease.release()
            raise
