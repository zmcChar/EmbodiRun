from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from embodied_runtime.distributed.communication import OpenPIProtocolError, validate_openpi_actions

from ._openpi_helpers import gr00t_actions


@pytest.mark.parametrize(
    "actions,match",
    [
        (
            np.zeros((1, 4, 2), dtype=np.float64),
            "dtype float32",
        ),
        (
            np.full((1, 4, 2), np.nan, dtype=np.float32),
            "non-finite",
        ),
        (
            {
                "arm": np.zeros((1, 4, 2), dtype=np.float32),
                "gripper": np.zeros((1, 5, 1), dtype=np.float32),
            },
            "share batch and horizon",
        ),
        (
            np.zeros((4,), dtype=np.float32),
            "must have shape",
        ),
    ],
)
def test_action_validation_rejects_unsafe_outputs(actions: Any, match: str) -> None:
    with pytest.raises(OpenPIProtocolError, match=match):
        validate_openpi_actions(actions)


def test_action_validation_accepts_unbatched_and_batched_common_shapes() -> None:
    unbatched = np.zeros((40, 17), dtype=np.float32)
    assert (
        validate_openpi_actions(
            unbatched,
            expected_batch_size=1,
            expected_action_horizon=40,
            expected_action_dim=17,
        )
        is unbatched
    )

    batched = gr00t_actions(batch=2)
    validated = validate_openpi_actions(
        batched,
        expected_batch_size=2,
        expected_action_horizon=40,
        expected_action_dim=17,
    )
    assert validated.keys() == batched.keys()
