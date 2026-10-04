from abstention_over_training_evals.adapters.olmo_settings import (
    OLMO3_THINK_MAX_TOKENS,
    OLMO3_THINK_TEMPERATURE,
    OLMO3_THINK_TOP_P,
)


def test_recommended_settings_match_the_model_card():
    assert OLMO3_THINK_TEMPERATURE == 0.6
    assert OLMO3_THINK_TOP_P == 0.95
    assert OLMO3_THINK_MAX_TOKENS == 32768
