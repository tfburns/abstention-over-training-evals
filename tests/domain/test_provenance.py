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


def _source(**overrides):
    base = dict(
        dataset="Transluce/WeirdChat",
        version="v1.0.1",
        license="MIT",
        url="https://weirdchat.transluce.org/",
        record_id="abc",
        metadata={"behavior_id": "purging-advice"},
    )
    base.update(overrides)
    return base


def test_dataset_provenance_normalizes_license_and_keeps_source():
    prov = Provenance(author_type="dataset", source=_source())
    assert prov.source is not None
    assert prov.source.license == "mit"
    assert prov.source.metadata["behavior_id"] == "purging-advice"


def test_dataset_provenance_requires_source():
    with pytest.raises(ValidationError, match="must record their source"):
        Provenance(author_type="dataset")


def test_dataset_provenance_rejects_non_permissive_license():
    with pytest.raises(ValidationError, match="not in the permissive allowlist"):
        Provenance(author_type="dataset", source=_source(license="cc-by-nc-4.0"))


def test_dataset_provenance_rejects_generator_fields():
    with pytest.raises(ValidationError, match="must not carry generator provenance"):
        Provenance(author_type="dataset", source=_source(), generator_model="x", generator_license="mit")


def test_non_dataset_provenance_rejects_source():
    with pytest.raises(ValidationError, match="only dataset prompts carry a source"):
        Provenance(author_type="human", source=_source())
