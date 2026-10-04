"""Developer command runners wired to ``uv run validate`` and ``uv run test``.

These exist so the quality gates have one entry point each, matching the project
standards in ``AGENTS.md``. They are not imported by the library.
"""

import subprocess
from collections.abc import Sequence

_VALIDATE_STEPS: tuple[tuple[str, ...], ...] = (
    ("ruff", "check", "src", "tests"),
    ("ruff", "format", "--check", "src", "tests"),
    ("mypy",),
    ("lint-imports",),
)

_TEST_STEPS: tuple[tuple[str, ...], ...] = (("pytest",),)


def _run(steps: Sequence[Sequence[str]]) -> int:
    for step in steps:
        # Fixed command lists, no shell.
        result = subprocess.run(step)
        if result.returncode != 0:
            return result.returncode
    return 0


def validate() -> None:
    """Run ruff check, ruff format check, mypy, and the import-linter contracts."""
    raise SystemExit(_run(_VALIDATE_STEPS))


def test() -> None:
    """Run the pytest suite."""
    raise SystemExit(_run(_TEST_STEPS))
