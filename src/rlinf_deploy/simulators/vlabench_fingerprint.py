"""Stable observation fingerprints for paired VLABench resets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def observation_fingerprint(values: Any) -> str:
    """Return a stable content digest for paired-reset validation."""

    digest = hashlib.sha256()
    _update_digest(digest, values)
    return digest.hexdigest()


def _update_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Mapping):
        digest.update(b"mapping:")
        for key in sorted(value, key=str):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"=")
            _update_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(f"sequence:{len(value)}:".encode())
        for item in value:
            _update_digest(digest, item)
        return

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    tobytes = getattr(value, "tobytes", None)
    if shape is not None and dtype is not None and callable(tobytes):
        digest.update(f"array:{tuple(shape)}:{dtype}:".encode())
        digest.update(tobytes())
        return
    if isinstance(value, bytes):
        digest.update(b"bytes:")
        digest.update(value)
        return
    digest.update(f"scalar:{type(value).__name__}:{value!r}".encode())


__all__ = ["observation_fingerprint"]
