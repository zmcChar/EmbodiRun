from __future__ import annotations

import pytest

from embodied_runtime.utils import DataUrlError, decode_data_url, encode_data_url


def test_data_url_round_trip_is_bounded() -> None:
    payload = b"\xff\xd8jpeg"
    encoded = encode_data_url(payload, "image/jpeg", maximum_bytes=16)
    assert decode_data_url(encoded, "image/jpeg", maximum_bytes=16) == payload


def test_data_url_rejects_wrong_media_and_oversized_input() -> None:
    value = encode_data_url(b"png", "image/png")
    with pytest.raises(DataUrlError, match="image/jpeg"):
        decode_data_url(value, "image/jpeg")
    with pytest.raises(DataUrlError, match="exceeds"):
        encode_data_url(b"123", "image/png", maximum_bytes=2)
