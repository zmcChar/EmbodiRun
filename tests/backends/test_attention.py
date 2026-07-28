from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from embodied_runtime.backends.torch_cuda.operators import (  # noqa: E402
    EagerAttention,
    SDPAAttention,
    get_attention_operator,
)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_sdpa_matches_eager_for_grouped_query_attention(kv_heads: int) -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(2, 4, 3, 8, generator=generator)
    key = torch.randn(2, kv_heads, 5, 8, generator=generator)
    value = torch.randn(2, kv_heads, 5, 8, generator=generator)
    mask = torch.zeros(2, 1, 3, 5)
    mask[:, :, :, -1] = -10_000

    expected = EagerAttention().attend(query, key, value, mask)
    actual = SDPAAttention().attend(query, key, value, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_attention_operator_error_suggests_registered_name() -> None:
    with pytest.raises(KeyError, match="did you mean"):
        get_attention_operator("sdp")
