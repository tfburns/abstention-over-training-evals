import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from abstention_over_training_evals.adapters.olmo_vllm import (  # noqa: E402
    OLMO3_THINK_TEMPERATURE,
    OLMO3_THINK_TOP_P,
    Olmo3Think32B,
)


def test_defaults_match_the_model_card():
    model = Olmo3Think32B(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    assert model._model_name == "allenai/Olmo-3-32B-Think"
    assert model._temperature == OLMO3_THINK_TEMPERATURE
    assert model._top_p == OLMO3_THINK_TOP_P


def test_overrides_are_respected():
    model = Olmo3Think32B(temperature=0.1, top_p=0.5, base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    assert model._temperature == 0.1
    assert model._top_p == 0.5
