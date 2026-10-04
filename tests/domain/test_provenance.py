import pytest
from pydantic import ValidationError

from abstention_over_training_evals.domain import AuthorType, Provenance


def test_human_provenance_needs_no_generator():
    prov = Provenance(author_type=AuthorType.HUMAN)
    assert prov.author_type is AuthorType.HUMAN
    assert prov.generator_model is None


def test_human_provenance_rejects_generator_fields():
    with pytest.raises(ValidationError):
        Provenance(author_type=AuthorType.HUMAN, generator_model="whatever")


def test_llm_provenance_requires_model_and_license():
    prov = Provenance(
        author_type=AuthorType.LLM,
        generator_model="OLMo-2-1124-7B-Instruct",
        generator_license="apache-2.0",
    )
    assert prov.generator_model == "OLMo-2-1124-7B-Instruct"


def test_llm_provenance_rejects_missing_model():
    with pytest.raises(ValidationError):
        Provenance(author_type=AuthorType.LLM, generator_license="apache-2.0")


def test_llm_provenance_rejects_missing_license():
    with pytest.raises(ValidationError):
        Provenance(author_type=AuthorType.LLM, generator_model="some-model")


def test_llm_provenance_rejects_non_permissive_license():
    with pytest.raises(ValidationError):
        Provenance(
            author_type=AuthorType.LLM,
            generator_model="some-model",
            generator_license="llama-community",
        )


def test_license_is_normalized_to_lowercase():
    prov = Provenance(
        author_type=AuthorType.LLM,
        generator_model="some-model",
        generator_license="Apache-2.0",
    )
    assert prov.generator_license == "apache-2.0"


def test_model_and_license_are_stripped():
    prov = Provenance(
        author_type=AuthorType.LLM,
        generator_model="  some-model  ",
        generator_license="  APACHE-2.0 ",
    )
    assert prov.generator_model == "some-model"
    assert prov.generator_license == "apache-2.0"


def test_whitespace_only_model_is_rejected():
    with pytest.raises(ValidationError):
        Provenance(author_type=AuthorType.LLM, generator_model="   ", generator_license="mit")
