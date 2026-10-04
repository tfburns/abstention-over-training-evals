import pytest

from abstention_over_training_evals.domain import (
    MAX_LOCAL_PROMPTS,
    LocalPromptBudgetError,
    enforce_local_prompt_budget,
)


def test_budget_allows_count_at_the_cap():
    enforce_local_prompt_budget(MAX_LOCAL_PROMPTS, source="file.jsonl")


def test_budget_allows_fewer():
    enforce_local_prompt_budget(0, source="file.jsonl")


def test_budget_rejects_over_the_cap():
    with pytest.raises(LocalPromptBudgetError) as exc:
        enforce_local_prompt_budget(MAX_LOCAL_PROMPTS + 1, source="file.jsonl")
    assert "file.jsonl" in str(exc.value)
    assert str(MAX_LOCAL_PROMPTS) in str(exc.value)
