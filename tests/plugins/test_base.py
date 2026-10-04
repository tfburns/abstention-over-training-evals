import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from abstention_over_training_evals.plugins.base import (  # noqa: E402
    LocalJsonlSafetyTask,
    resolve_prompts_path,
)


def test_per_task_override_is_used(monkeypatch):
    monkeypatch.setenv("AOTE_PROMPTS_PATH__MyTask", "/tmp/mytask.jsonl")
    assert resolve_prompts_path("MyTask", "default.jsonl").name == "mytask.jsonl"


def test_override_is_scoped_to_its_task(monkeypatch):
    # An override for one task must not redirect another.
    monkeypatch.setenv("AOTE_PROMPTS_PATH__OtherTask", "/tmp/other.jsonl")
    monkeypatch.delenv("AOTE_PROMPTS_PATH__MyTask", raising=False)
    resolved = resolve_prompts_path("MyTask", "default.jsonl")
    assert resolved.name == "default.jsonl"
    assert resolved.parent.name == "local"


def test_default_path_uses_the_filename(monkeypatch):
    monkeypatch.delenv("AOTE_PROMPTS_PATH__MyTask", raising=False)
    resolved = resolve_prompts_path("MyTask", "default.jsonl")
    assert resolved.name == "default.jsonl"
    assert resolved.parent.name == "local"


def test_missing_prompts_filename_is_rejected():
    class NoFilename(LocalJsonlSafetyTask):
        NAME = "NoFilename"

    with pytest.raises(ValueError, match="PROMPTS_FILENAME"):
        NoFilename()
