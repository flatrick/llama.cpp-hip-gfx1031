"""CLI entry point. Phase 1 exposes only `migrate`; the dashboard is Phase 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llamactl.core.migrate import migrate

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llamactl")
    sub = parser.add_subparsers(dest="command")
    mig = sub.add_parser(
        "migrate", help="One-shot import of models/*.json into configs/models/*.toml"
    )
    mig.add_argument("--src", type=Path, default=REPO_ROOT / "models")
    mig.add_argument("--dest", type=Path, default=REPO_ROOT / "configs" / "models")
    mig.add_argument(
        "--force", action="store_true", help="Overwrite existing .toml files"
    )
    args = parser.parse_args(argv)

    if args.command != "migrate":
        parser.print_help()
        print("\nThe dashboard arrives in Phase 2; only 'migrate' exists today.")
        return 0

    failed = False
    for dest, status, warnings in migrate(args.src, args.dest, force=args.force):
        print(f"{status:>8}  {dest}")
        for warning in warnings:
            print(f"          WARNING: {warning}")
        failed = failed or status == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
