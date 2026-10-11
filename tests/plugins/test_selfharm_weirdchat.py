import json

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.adapters import JsonlPromptStore  # noqa: E402
from abstention_over_training_evals.domain import AuthorType  # noqa: E402
from abstention_over_training_evals.plugins.base import _DATA_DIR  # noqa: E402
from abstention_over_training_evals.plugins.selfharm_weirdchat.selfharm_weirdchat import (  # noqa: E402
    SelfHarmWeirdChat,
    register_tasks,
)

_ROW = {
    "id": "weirdchat-abc",
    "prompt": "a single-turn prompt",
    "provenance": {
        "author_type": "dataset",
        "source": {
            "dataset": "Transluce/WeirdChat",
            "version": "v1.0.1",
            "license": "mit",
            "url": "https://weirdchat.transluce.org/",
            "record_id": "abc",
            "metadata": {"behavior_id": "purging-advice"},
        },
    },
}


@pytest.fixture
def prompts_file(tmp_path, monkeypatch):
    f = tmp_path / "prompts.jsonl"
    f.write_text(json.dumps(_ROW) + "\n", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__SelfHarmWeirdChat", str(f))
    return f


def test_task_yields_single_turn_samples(prompts_file):
    task = SelfHarmWeirdChat.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    samples = list(task.iterate_samples())
    assert len(samples) == 1
    assert samples[0].messages[-1].content == "a single-turn prompt"


def test_metadata_lists_source_dataset(prompts_file):
    task = SelfHarmWeirdChat.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    meta = task.get_metadata()
    assert meta["prompt_author_types"] == ["dataset"]
    assert meta["prompt_source_datasets"] == ["Transluce/WeirdChat@v1.0.1 (mit)"]


def test_shipped_prompts_all_come_from_weirdchat():
    items = JsonlPromptStore(_DATA_DIR / SelfHarmWeirdChat.PROMPTS_FILENAME).load()
    assert items
    assert all(i.provenance.author_type is AuthorType.DATASET for i in items)
    assert all(i.provenance.source and i.provenance.source.dataset == "Transluce/WeirdChat" for i in items)


def test_register_tasks_adds_the_task():
    registry = Registry()
    register_tasks(registry)
    assert "SelfHarmWeirdChat" in registry
