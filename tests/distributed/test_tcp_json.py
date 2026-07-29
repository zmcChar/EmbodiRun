from __future__ import annotations

import asyncio
import json
import struct
from functools import wraps

import pytest

from embodied_runtime.distributed.communication import (
    TcpJsonProtocolError,
    TcpJsonRequestClient,
    read_json_message,
    write_json_message,
)


def async_test(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return wrapper


@async_test
async def test_tcp_json_client_round_trips_one_request() -> None:
    handler_closed = asyncio.Event()

    async def echo(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await read_json_message(reader)
            await write_json_message(
                writer,
                {
                    "request_id": request["request_id"],
                    "actions": [[1.0, 2.0]],
                },
            )
        finally:
            writer.close()
            await writer.wait_closed()
            handler_closed.set()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        response = await TcpJsonRequestClient("127.0.0.1", port).request(
            {"request_id": "request-1"}
        )
    finally:
        server.close()
        await server.wait_closed()
        await asyncio.wait_for(handler_closed.wait(), timeout=1.0)

    assert response == {
        "request_id": "request-1",
        "actions": [[1.0, 2.0]],
    }


@async_test
async def test_tcp_json_writer_rejects_oversized_payload() -> None:
    class UnusedWriter:
        def write(self, _data: bytes) -> None:
            raise AssertionError("oversized messages must be rejected before writing")

        async def drain(self) -> None:
            raise AssertionError("oversized messages must be rejected before draining")

    with pytest.raises(TcpJsonProtocolError, match="exceeds limit"):
        await write_json_message(
            UnusedWriter(),
            {"payload": "too-large"},
            max_message_bytes=4,
        )


@pytest.mark.parametrize(
    "encoded,match",
    [
        (json.dumps([1, 2, 3]).encode(), "top-level JSON message must be an object"),
        (b'{"value":NaN}', "non-standard JSON constant"),
        (b"not-json", "invalid JSON message"),
    ],
)
@async_test
async def test_tcp_json_reader_rejects_malformed_payloads(
    encoded: bytes,
    match: str,
) -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", len(encoded)) + encoded)
    reader.feed_eof()

    with pytest.raises(TcpJsonProtocolError, match=match):
        await read_json_message(reader)


@async_test
async def test_tcp_json_reader_rejects_oversized_header_before_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(struct.pack("!I", 1024))
    reader.feed_eof()

    with pytest.raises(TcpJsonProtocolError, match="exceeds limit"):
        await read_json_message(reader, max_message_bytes=16)


def test_tcp_json_client_validates_message_limit_before_connecting() -> None:
    with pytest.raises(ValueError, match="max_message_bytes"):
        TcpJsonRequestClient("127.0.0.1", 1234, max_message_bytes=0)
