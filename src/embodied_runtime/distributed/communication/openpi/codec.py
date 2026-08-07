"""Optional OpenPI msgpack codec loading and exchange normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import OpenPIDependencyError, OpenPIError, OpenPIProtocolError
from .protocols import OpenPIMessageCodec


class OpenPIClientCodec:
    """Adapter around the codec shipped by ``openpi-client``."""

    def __init__(self) -> None:
        try:
            from openpi_client import msgpack_numpy
        except ImportError as error:
            raise OpenPIDependencyError(
                "OpenPI WebSocket serialization requires the optional `openpi-client` package"
            ) from error
        self._module = msgpack_numpy
        self._packer = msgpack_numpy.Packer()

    def pack(self, value: Any) -> bytes:
        return self._packer.pack(value)

    def unpack(self, value: bytes | str) -> Any:
        if isinstance(value, str):
            # OpenPI inference and handshake frames are msgpack, while some
            # compatible servers acknowledge reset with a plain text frame.
            return value
        return self._module.unpackb(value)


def pack_request(codec: OpenPIMessageCodec, payload: Mapping[str, Any]) -> bytes:
    try:
        packed = codec.pack(dict(payload))
    except OpenPIError:
        raise
    except Exception as error:
        raise OpenPIProtocolError(f"failed to encode OpenPI request: {error}") from error
    if not isinstance(packed, bytes):
        raise OpenPIProtocolError(
            f"OpenPI codec.pack must return bytes, got {type(packed).__name__}"
        )
    return packed


def unpack_response(codec: OpenPIMessageCodec, response: bytes | str) -> Any:
    try:
        return codec.unpack(response)
    except OpenPIError:
        raise
    except Exception as error:
        raise OpenPIProtocolError(f"failed to decode OpenPI response: {error}") from error
