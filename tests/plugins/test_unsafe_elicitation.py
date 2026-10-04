from types import SimpleNamespace

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.shared.types import Error  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.domain import SafetyJudgement  # noqa: E402
from abstention_over_training_evals.plugins.unsafe_elicitation.unsafe_elicitation import (  # noqa: E402
    SafetyJudgeMetric,
    UnsafeElicitation,
    register_tasks,
)


@pytest.fixture
def prompts_file(tmp_path, monkeypatch):
    f = tmp_path / "prompts.jsonl"
    f.write_text(
        '{"id": "a", "prompt": "a single-turn prompt", "provenance": {"author_type": "human"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__UnsafeElicitation", str(f))
    return f


@pytest.fixture
def mixed_prompts_file(tmp_path, monkeypatch):
    f = tmp_path / "prompts.jsonl"
    f.write_text(
        '{"id": "h", "prompt": "human one", "provenance": {"author_type": "human"}}\n'
        '{"id": "l", "prompt": "llm one", "provenance": {"author_type": "llm", '
        '"generator_model": "OLMo-2-1124-7B-Instruct", "generator_license": "apache-2.0"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__UnsafeElicitation", str(f))
    return f


def _completion(**overrides):
    base = dict(error=None, prompt="p", completion="answer", raw_completion_reasoning="trace")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_task_yields_single_turn_samples_without_ground_truth(prompts_file):
    task = UnsafeElicitation.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    samples = list(task.iterate_samples())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.ground_truth is None
    assert sample.messages[-1].content == "a single-turn prompt"
    assert sum(1 for m in sample.messages if m.role.name == "USER") == 1


def test_metadata_includes_provenance_summary(mixed_prompts_file):
    task = UnsafeElicitation.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    meta = task.get_metadata()
    assert meta["dataset_format"] == "local-jsonl"
    assert meta["prompt_count"] == "2"
    assert set(meta["prompt_author_types"]) == {"human", "llm"}
    assert "OLMo-2-1124-7B-Instruct" in meta["prompt_generator_models"]
    assert "apache-2.0" in meta["prompt_generator_licenses"]
    assert str(meta["dataset_path"]).endswith("prompts.jsonl")


def test_sample_rows_carry_provenance(mixed_prompts_file):
    task = UnsafeElicitation.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    task._load_dataset("default")
    rows = task.dataset[task.SAMPLE_SPLIT]
    assert rows[1]["provenance"]["author_type"] == "llm"


def test_empty_dataset_raises_an_actionable_error(tmp_path, monkeypatch):
    f = tmp_path / "prompts.jsonl"
    f.write_text("", encoding="utf-8")
    monkeypatch.setenv("AOTE_PROMPTS_PATH__UnsafeElicitation", str(f))
    task = UnsafeElicitation.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    with pytest.raises(ValueError, match="contains no prompts"):
        task._load_dataset("default")


def test_register_tasks_adds_the_task():
    registry = Registry()
    register_tasks(registry)
    assert "UnsafeElicitation" in registry


def test_metric_abstains_with_the_default_judge(monkeypatch):
    monkeypatch.delenv("AOTE_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("AOTE_JUDGE_BASE_URL", raising=False)
    result = SafetyJudgeMetric().calculate(_completion())[0]
    assert result.value is None


def test_metric_scores_unsafe_as_one():
    class _Unsafe:
        def judge(self, prompt, answer, reasoning):
            return SafetyJudgement(judged=True, unsafe=True, reason="bad")

    result = SafetyJudgeMetric(judge=_Unsafe()).calculate(_completion())[0]
    assert result.value == 1.0
    assert result.higher_is_better is False


def test_metric_propagates_response_error():
    err = Error(error_class="BoomError", message="boom", traceback="")
    result = SafetyJudgeMetric().calculate(_completion(error=err))[0]
    assert result.value is None
    assert result.error is err
