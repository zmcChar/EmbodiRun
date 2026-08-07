"""CLI entrypoint for language-conditioned navigation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

from .navigation.cli import apply_cli_overrides, build_parser
from .navigation.config import load_config
from .navigation.runner import run_navigation


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = apply_cli_overrides(load_config(args.config), args)
        asyncio.run(run_navigation(config, execute=args.execute))
    except KeyboardInterrupt:
        print(json.dumps({"kind": "navigation_summary", "ok": False, "reason": "interrupted"}))
        return 130
    except Exception as error:  # noqa: BLE001 - command boundary reports structured failure
        print(
            json.dumps(
                {
                    "kind": "navigation_summary",
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
