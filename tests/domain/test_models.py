import pytest
from pydantic import ValidationError

from abstention_over_training_evals.domain import AuthorType, PromptItem, Provenance

_HUMAN = Provenance(author_type=AuthorType.HUMAN)


def test_prompt_item_round_trips():
    item = PromptItem(id="a1", prompt="hello", provenance=_HUMAN)
    assert item.id == "a1"
    assert item.prompt == "hello"
    assert item.provenance.author_type is AuthorType.HUMAN


def test_prompt_item_is_frozen():
    item = PromptItem(id="a1", prompt="hello", provenance=_HUMAN)
    with pytest.raises(ValidationError):
        item.prompt = "changed"


def test_prompt_item_requires_provenance():
    with pytest.raises(ValidationError):
        PromptItem.model_validate({"id": "a1", "prompt": "hello"})


def test_prompt_item_parses_nested_llm_provenance():
    item = PromptItem.model_validate(
        {
            "id": "a1",
            "prompt": "hello",
            "provenance": {
                "author_type": "llm",
                "generator_model": "OLMo-2-1124-7B-Instruct",
                "generator_license": "apache-2.0",
            },
        }
    )
    assert item.provenance.author_type is AuthorType.LLM


def test_prompt_item_rejects_blank_fields():
    with pytest.raises(ValidationError):
        PromptItem(id="a1", prompt="   ", provenance=_HUMAN)
    with pytest.raises(ValidationError):
        PromptItem(id="", prompt="hello", provenance=_HUMAN)


def test_prompt_item_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PromptItem.model_validate(
            {"id": "a1", "prompt": "hello", "provenance": {"author_type": "human"}, "answer": "nope"}
        )
