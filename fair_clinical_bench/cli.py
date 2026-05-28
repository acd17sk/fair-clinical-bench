"""Command-line entry point. Real subcommands wired by the swarm during build."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Stub. Replaced during Phase 4 of the build (CLI wiring task)."""
    args = argv if argv is not None else sys.argv[1:]
    print("fair-clinical-bench: not yet implemented. Run `/swarm plan` to build.")
    if args:
        print(f"  (received args: {args!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

