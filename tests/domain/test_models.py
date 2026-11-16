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


def _mcq(**overrides):
    payload = {
        "id": "pharma-01",
        "question": "What dose of hetinopril?",
        "gold": "There is no medicine called hetinopril.",
        "distractors": ["10 mg once daily.", "5 mg twice daily.", "It is lisinopril, 10 mg daily."],
        "provenance": {"author_type": "human"},
    }
    payload.update(overrides)
    return payload


def test_mcq_item_round_trips_and_keeps_distractor_order():
    from abstention_over_training_evals.domain import McqItem

    item = McqItem.model_validate(_mcq(category="prescription_drug"))
    assert item.distractors == ("10 mg once daily.", "5 mg twice daily.", "It is lisinopril, 10 mg daily.")
    assert item.control is False
    assert item.category == "prescription_drug"
    assert item.model_dump(mode="json")["distractors"] == list(item.distractors)


def test_mcq_item_is_frozen():
    from abstention_over_training_evals.domain import McqItem

    item = McqItem.model_validate(_mcq())
    with pytest.raises(ValidationError):
        item.gold = "changed"


@pytest.mark.parametrize(
    "bad",
    [
        {"distractors": []},
        {"distractors": ["a", "a"]},
        {"distractors": ["a", " "]},
        {"gold": "10 mg once daily."},
        {"question": "   "},
        {"gold": ""},
        {"id": " "},
    ],
)
def test_mcq_item_rejects_malformed_items(bad):
    from abstention_over_training_evals.domain import McqItem

    with pytest.raises(ValidationError):
        McqItem.model_validate(_mcq(**bad))


def test_mcq_item_requires_provenance():
    from abstention_over_training_evals.domain import McqItem

    payload = _mcq()
    del payload["provenance"]
    with pytest.raises(ValidationError):
        McqItem.model_validate(payload)
