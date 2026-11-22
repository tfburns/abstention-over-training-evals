"""Reads local JSONL into validated :class:`PromptItem` or :class:`McqItem` objects."""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from abstention_over_training_evals.domain import (
    McqItem,
    PromptItem,
    enforce_local_prompt_budget,
)


class JsonlPromptStore:
    """Loads prompts from a local JSONL file, one object per line.

    Each line is a ``{"id", "prompt", "provenance", "category"}`` object (see
    :class:`PromptItem`). The category is optional for tasks that do not use categories. The store
    validates each row, rejects duplicate ids, and enforces the local-prompt budget.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[PromptItem]:
        """Read and validate every prompt in the file.

        Returns:
            The prompts in file order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a line is not a JSON object or an id is duplicated.
            LocalPromptBudgetError: If the file holds more prompts than the budget allows.
        """
        if not self._path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {self._path}")

        items: list[PromptItem] = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = self._parse_line(raw, line_number)
            if item.id in seen_ids:
                raise ValueError(f"{self._path}:{line_number}: duplicate id {item.id!r}")
            seen_ids.add(item.id)
            items.append(item)

        enforce_local_prompt_budget(len(items), source=str(self._path))
        logger.debug("Loaded {} prompts from {}", len(items), self._path)
        return items

    def _parse_line(self, raw: str, line_number: int) -> PromptItem:
        return PromptItem.model_validate(_parse_json_object(self._path, raw, line_number))


class JsonlMcqStore:
    """Loads multiple-choice items from a local JSONL file, one object per line.

    Each line is a ``{"id", "question", "gold", "distractors", "provenance", "category", "control"}``
    object (see :class:`McqItem`); ``category`` and ``control`` are optional. The store validates each
    row, rejects duplicate ids, and enforces the local-prompt budget.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[McqItem]:
        """Read and validate every item in the file.

        Returns:
            The items in file order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If a line is not a JSON object, an item is malformed, or an id is duplicated.
            LocalPromptBudgetError: If the file holds more items than the budget allows.
        """
        if not self._path.is_file():
            raise FileNotFoundError(f"Item file not found: {self._path}")

        items: list[McqItem] = []
        seen_ids: set[str] = set()
        for line_number, raw in enumerate(self._path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            item = McqItem.model_validate(_parse_json_object(self._path, raw, line_number))
            if item.id in seen_ids:
                raise ValueError(f"{self._path}:{line_number}: duplicate id {item.id!r}")
            seen_ids.add(item.id)
            items.append(item)

        enforce_local_prompt_budget(len(items), source=str(self._path))
        logger.debug("Loaded {} multiple-choice items from {}", len(items), self._path)
        return items


def _parse_json_object(path: Path, raw: str, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}:{line_number}: expected a JSON object, got {type(payload).__name__}")
    return payload
