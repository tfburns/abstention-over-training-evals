from types import SimpleNamespace

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.adapters.llm_judge import (  # noqa: E402
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    LlmSafetyJudge,
)
from abstention_over_training_evals.domain import SafetyJudgement  # noqa: E402
from abstention_over_training_evals.plugins.nonexistent_entities.nonexistent_entities import (  # noqa: E402
    FABRICATION_RUBRIC,
    OVER_REFUSAL_RUBRIC,
    FabricationJudgeMetric,
    NonexistentEntities,
    NonexistentEntitiesControls,
    OverRefusalJudgeMetric,
    register_tasks,
)


@pytest.fixture
def prompts_file(tmp_path, monkeypatch):
    f = tmp_path / "prompts.jsonl"
    f.write_text(
        '{"id": "pharma-01", "prompt": "What dose of a made-up drug?", "provenance": {"author_type": "human"}, '
        '"category": "prescription_drug"}\n'
        '{"id": "elec-01", "prompt": "What does a made-up article require?", "provenance": {"author_type": "human"}, '
        '"category": "electrical_code"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__NonexistentEntities", str(f))
    return f


@pytest.fixture
def controls_file(tmp_path, monkeypatch):
    f = tmp_path / "controls.jsonl"
    f.write_text(
        '{"id": "ctrl-pharma-01", "prompt": "A question about a real drug", "provenance": {"author_type": "human"}}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__NonexistentEntitiesControls", str(f))
    return f


def _completion(**overrides):
    base = dict(error=None, prompt="p", completion="answer", raw_completion_reasoning="trace")
    base.update(overrides)
    return SimpleNamespace(**base)


def test_task_yields_single_turn_samples_without_ground_truth(prompts_file):
    task = NonexistentEntities.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    samples = list(task.iterate_samples())
    assert len(samples) == 2
    assert all(s.ground_truth is None for s in samples)
    assert samples[0].messages[-1].content == "What dose of a made-up drug?"
    assert sum(1 for m in samples[0].messages if m.role.name == "USER") == 1


def test_controls_task_reads_its_own_file(controls_file, prompts_file):
    # The two tasks must not share a prompt file: the fabrication rubric assumes every entity is fake.
    controls = NonexistentEntitiesControls.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    fakes = NonexistentEntities.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    assert [s.messages[-1].content for s in controls.iterate_samples()] == ["A question about a real drug"]
    assert len(list(fakes.iterate_samples())) == 2


def test_metadata_reports_human_provenance(prompts_file):
    task = NonexistentEntities.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    meta = task.get_metadata()
    assert meta["dataset_format"] == "local-jsonl"
    assert meta["prompt_count"] == "2"
    assert meta["prompt_author_types"] == ["human"]
    assert meta["prompt_categories"] == ["electrical_code", "prescription_drug"]


def test_register_tasks_adds_both_tasks():
    registry = Registry()
    register_tasks(registry)
    assert "NonexistentEntities" in registry
    assert "NonexistentEntitiesControls" in registry


def test_tasks_use_their_own_metrics():
    assert NonexistentEntities.METRICS == [FabricationJudgeMetric]
    assert NonexistentEntitiesControls.METRICS == [OverRefusalJudgeMetric]
    assert FabricationJudgeMetric.NAME == "Fabrication Rate"
    assert OverRefusalJudgeMetric.NAME == "Over-refusal Rate"


def test_fabrication_rubric_states_the_decisive_rules():
    # The rubric is what a reviewer reads to understand the number; keep its load-bearing clauses.
    for clause in ("does NOT exist", "hedge that is then overridden", "Equating the entity with a real one", "SAFE"):
        assert clause in FABRICATION_RUBRIC
    assert "over-refusal" in OVER_REFUSAL_RUBRIC


def test_fabrication_metric_configures_the_judge_with_its_rubric(monkeypatch):
    monkeypatch.setenv(JUDGE_MODEL_ENV, "judge-model")
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://127.0.0.1:8001/v1")
    judge = FabricationJudgeMetric()._judge
    assert isinstance(judge, LlmSafetyJudge)
    assert judge._system_prompt == FABRICATION_RUBRIC
    # The verdict is about what the user acts on; the trace is analysed separately.
    assert judge._include_reasoning is False


def test_metric_abstains_with_the_default_judge(monkeypatch):
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)
    assert FabricationJudgeMetric().calculate(_completion())[0].value is None


def test_metric_scores_fabrication_as_one():
    class _Fabricated:
        def judge(self, prompt, answer, reasoning):
            return SafetyJudgement(judged=True, unsafe=True, reason="asserted a dose for a fake drug")

    result = FabricationJudgeMetric(judge=_Fabricated()).calculate(_completion())[0]
    assert result.value == 1.0
    assert result.higher_is_better is False
    assert result.metric_name == "Fabrication Rate"
