import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from abstention_over_training_evals.adapters import JsonlPromptStore
from abstention_over_training_evals.domain import MAX_LOCAL_PROMPTS, LocalPromptBudgetError

_HUMAN = {"author_type": "human"}


def _line(id_: str, prompt: str, provenance: dict | None = None) -> str:
    return json.dumps({"id": id_, "prompt": prompt, "provenance": provenance or _HUMAN})


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_prompts_in_order(tmp_path):
    f = _write(tmp_path / "p.jsonl", [_line("a", "one"), _line("b", "two")])
    items = JsonlPromptStore(f).load()
    assert [i.id for i in items] == ["a", "b"]
    assert items[0].prompt == "one"


def test_loads_llm_provenance(tmp_path):
    prov = {
        "author_type": "llm",
        "generator_model": "OLMo-2-1124-7B-Instruct",
        "generator_license": "apache-2.0",
    }
    f = _write(tmp_path / "p.jsonl", [_line("a", "one", prov)])
    item = JsonlPromptStore(f).load()[0]
    assert item.provenance.generator_model == "OLMo-2-1124-7B-Instruct"


def test_rejects_non_permissive_llm_provenance(tmp_path):
    prov = {"author_type": "llm", "generator_model": "x", "generator_license": "llama-community"}
    f = _write(tmp_path / "p.jsonl", [_line("a", "one", prov)])
    with pytest.raises(ValidationError):
        JsonlPromptStore(f).load()


def test_blank_lines_are_skipped(tmp_path):
    f = _write(tmp_path / "p.jsonl", [_line("a", "one"), "", "   "])
    assert len(JsonlPromptStore(f).load()) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        JsonlPromptStore(tmp_path / "missing.jsonl").load()


def test_invalid_json_raises(tmp_path):
    f = _write(tmp_path / "p.jsonl", ["{not json}"])
    with pytest.raises(ValueError):
        JsonlPromptStore(f).load()


def test_non_object_line_raises(tmp_path):
    f = _write(tmp_path / "p.jsonl", ["[1, 2, 3]"])
    with pytest.raises(ValueError):
        JsonlPromptStore(f).load()


def test_duplicate_id_raises(tmp_path):
    f = _write(tmp_path / "p.jsonl", [_line("a", "one"), _line("a", "two")])
    with pytest.raises(ValueError):
        JsonlPromptStore(f).load()


def test_budget_is_enforced(tmp_path):
    lines = [_line(str(n), f"p{n}") for n in range(MAX_LOCAL_PROMPTS + 1)]
    f = _write(tmp_path / "p.jsonl", lines)
    with pytest.raises(LocalPromptBudgetError):
        JsonlPromptStore(f).load()
