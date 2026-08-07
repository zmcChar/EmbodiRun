"""Process-local ownership and dtype policy for mutable PyTorch modules."""

from __future__ import annotations

from threading import Lock
from typing import Any
from weakref import ref

from ..errors import UnsupportedBackendError

_ACTIVE_MODULES: dict[int, tuple[ref[Any], object]] = {}
_MODULE_DTYPE_POLICIES: dict[int, tuple[ref[Any], str]] = {}
_MODULES_LOCK = Lock()


class RuntimeModuleLease:
    """Exclusive live-session ownership with a commit-on-success policy."""

    def __init__(
        self,
        *,
        module_key: int | None = None,
        module_ref: ref[Any] | None = None,
        token: object | None = None,
        dtype_policy: str = "preserve",
    ) -> None:
        self._module_key = module_key
        self._module_ref = module_ref
        self._token = token
        self._dtype_policy = dtype_policy
        self._released = False

    def commit_dtype_policy(self) -> None:
        """Persist the policy only after the caller has completed ``load``."""

        if self._module_key is None or self._module_ref is None:
            return
        with _MODULES_LOCK:
            active = _ACTIVE_MODULES.get(self._module_key)
            if active is None or active[1] is not self._token:
                raise RuntimeError("cannot commit dtype policy without an active module lease")
            existing = _MODULE_DTYPE_POLICIES.get(self._module_key)
            if (
                existing is not None
                and existing[0]() is self._module_ref()
                and existing[1] != self._dtype_policy
            ):
                raise UnsupportedBackendError(
                    _dtype_policy_conflict(existing[1], self._dtype_policy)
                )
            _MODULE_DTYPE_POLICIES[self._module_key] = (
                self._module_ref,
                self._dtype_policy,
            )

    def release(self) -> None:
        if self._released or self._module_key is None:
            return
        with _MODULES_LOCK:
            active = _ACTIVE_MODULES.get(self._module_key)
            if active is not None and active[1] is self._token:
                _ACTIVE_MODULES.pop(self._module_key, None)
        self._released = True


def acquire_runtime_module(
    module: Any | None,
    *,
    dtype_policy: str,
) -> RuntimeModuleLease:
    """Acquire exclusive ownership and validate a module's persistent dtype.

    ``module.to(dtype=...)`` mutates the package-owned object.  The first
    successful load therefore fixes either ``preserve`` or one concrete dtype
    for that module object's lifetime.  Acquisition only validates the policy;
    the caller explicitly commits it after device placement and session
    construction succeed.
    """

    if module is None:
        return RuntimeModuleLease(dtype_policy=dtype_policy)

    module_key = id(module)
    token = object()

    def discard_dead_module(dead_ref: ref[Any]) -> None:
        with _MODULES_LOCK:
            active = _ACTIVE_MODULES.get(module_key)
            if active is not None and active[0] is dead_ref and active[1] is token:
                _ACTIVE_MODULES.pop(module_key, None)
            policy = _MODULE_DTYPE_POLICIES.get(module_key)
            if policy is not None and policy[0] is dead_ref:
                _MODULE_DTYPE_POLICIES.pop(module_key, None)

    module_ref = ref(module, discard_dead_module)
    with _MODULES_LOCK:
        active = _ACTIVE_MODULES.get(module_key)
        if active is not None and active[0]() is module:
            raise UnsupportedBackendError(
                "runtime_module is already bound to an active backend session; "
                "close that session before loading this module again"
            )
        if active is not None and active[0]() is None:
            _ACTIVE_MODULES.pop(module_key, None)

        policy = _MODULE_DTYPE_POLICIES.get(module_key)
        if policy is not None and policy[0]() is None:
            _MODULE_DTYPE_POLICIES.pop(module_key, None)
            policy = None
        if policy is not None and policy[0]() is module and policy[1] != dtype_policy:
            raise UnsupportedBackendError(_dtype_policy_conflict(policy[1], dtype_policy))

        _ACTIVE_MODULES[module_key] = (module_ref, token)

    return RuntimeModuleLease(
        module_key=module_key,
        module_ref=module_ref,
        token=token,
        dtype_policy=dtype_policy,
    )


def _dtype_policy_conflict(existing: str, requested: str) -> str:
    return (
        "runtime_module dtype policy is already locked to "
        f"{existing!r}; requested incompatible policy {requested!r}. "
        "Build a separate ModelPackage with a distinct module replica."
    )
