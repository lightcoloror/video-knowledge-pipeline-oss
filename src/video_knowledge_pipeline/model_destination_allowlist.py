from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model_api_settings import configured_remote_destination_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List exact HTTPS hosts used by enabled remote model profiles."
    )
    parser.add_argument("--settings-path", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = configured_remote_destination_status(
        Path(args.settings_path) if args.settings_path else None
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
