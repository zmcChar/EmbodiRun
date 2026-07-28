from __future__ import annotations

import pytest

from embodied_runtime.apps.local_pi05 import build_parser, main


def test_pi05_cli_exposes_cross_group_runtime_controls() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "/models/pi05",
            "--device",
            "cuda:1",
            "--dtype",
            "float16",
            "--num-steps",
            "3",
            "--mode",
            "compile",
        ]
    )

    assert args.checkpoint == "/models/pi05"
    assert args.device == "cuda:1"
    assert args.dtype == "float16"
    assert args.num_steps == 3
    assert args.mode == "compile"
    assert args.cuda_graph is False
    assert args.allow_download is False


def test_pi05_cli_preserves_checkpoint_dtypes_by_default() -> None:
    args = build_parser().parse_args(["--checkpoint", "/models/pi05", "--cuda-graph"])
    assert args.dtype == "preserve"
    assert args.cuda_graph is True


def test_pi05_cli_rejects_cuda_graph_before_loading_checkpoint() -> None:
    with pytest.raises(SystemExit, match="requires --mode eager"):
        main(
            [
                "--checkpoint",
                "/does/not/exist",
                "--mode",
                "compile",
                "--cuda-graph",
            ]
        )
    with pytest.raises(SystemExit, match="requires a cuda:N device"):
        main(
            [
                "--checkpoint",
                "/does/not/exist",
                "--device",
                "cpu",
                "--cuda-graph",
            ]
        )


def test_pi05_cli_rejects_prebatched_single_request() -> None:
    with pytest.raises(SystemExit, match="currently must be 1"):
        main(["--checkpoint", "/models/pi05", "--batch-size", "2"])
