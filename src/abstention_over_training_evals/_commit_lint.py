"""Validate a commit message against Conventional Commits.

Used by the ``commit-msg`` git hook and by CI. Only the header (first line) is checked:

    <type>(<optional scope>)<optional !>: <description>

Allowed types: build, chore, ci, docs, feat, fix, perf, refactor, revert, style, test. Auto-generated
``Merge``/``Revert`` headers and ``fixup!``/``squash!`` autosquash headers pass unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test")
MAX_HEADER_LENGTH = 100

_HEADER = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([\w .\-/]+\))?!?: .+")
_SKIP_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ")


def check(header: str) -> str | None:
    """Return an error message if the header is invalid, else ``None``."""
    if not header:
        return "empty commit message"
    if header.startswith(_SKIP_PREFIXES):
        return None
    if len(header) > MAX_HEADER_LENGTH:
        return f"header is longer than {MAX_HEADER_LENGTH} characters"
    if not _HEADER.match(header):
        return "header must match '<type>(<optional scope>)!?: <description>'"
    return None


def _header_from(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            return line.rstrip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a commit message against Conventional Commits.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--from-file", help="read the message from this file (git commit-msg hook passes it)")
    source.add_argument("--subject", help="validate this literal header")
    args = parser.parse_args()

    if args.subject is not None:
        header = _header_from(args.subject)
    elif args.from_file is not None:
        header = _header_from(Path(args.from_file).read_text(encoding="utf-8"))
    else:
        header = _header_from(sys.stdin.read())

    error = check(header)
    if error is None:
        return 0
    sys.stderr.write(
        f"Invalid commit message: {error}\n\n"
        f"  header: {header!r}\n\n"
        "Use Conventional Commits, for example:\n"
        "  feat: add a rate-limit metric\n"
        "  fix(judge): abstain on timeout\n"
        f"Allowed types: {', '.join(TYPES)}.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
