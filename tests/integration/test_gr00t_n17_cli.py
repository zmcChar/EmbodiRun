from __future__ import annotations

import pytest

from embodied_runtime.apps.gr00t_n17 import build_parser, main


def test_gr00t_cli_exposes_only_hf_provider() -> None:
    parser = build_parser()

    args = parser.parse_args(["--provider", "hf"])
    assert args.provider == "hf"

    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "not-a-provider"])


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--provider", "hf", "--timeout-s", "0"], "timeout"),
        (["--provider", "hf", "--image-height", "0"], "image-height"),
        (["--provider", "hf", "--prompt", ""], "prompt"),
    ],
)
def test_gr00t_cli_rejects_invalid_values_before_loading_weights(
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(SystemExit, match=message):
        main(arguments)
