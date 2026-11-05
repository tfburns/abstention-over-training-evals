from types import SimpleNamespace

import pytest

pytest.importorskip("openai", reason="requires the `eval` extra (openai client)")

from abstention_over_training_evals.adapters.llm_judge import (  # noqa: E402
    JUDGE_BASE_URL_ENV,
    JUDGE_MODEL_ENV,
    LlmSafetyJudge,
    parse_verdict,
    safety_judge_from_env,
)
from abstention_over_training_evals.adapters.unconfigured_judge import UnconfiguredSafetyJudge  # noqa: E402


def test_parse_verdict_unsafe_true():
    verdict = parse_verdict('{"unsafe": true, "reason": "gave bomb steps"}')
    assert verdict.judged is True
    assert verdict.unsafe is True
    assert verdict.reason == "gave bomb steps"


def test_parse_verdict_unsafe_false():
    verdict = parse_verdict('{"unsafe": false, "reason": "refused"}')
    assert verdict.judged is True
    assert verdict.unsafe is False


def test_parse_verdict_extracts_json_from_surrounding_prose():
    verdict = parse_verdict('Sure!\n{"unsafe": true, "reason": "x"}\nThanks')
    assert verdict.unsafe is True


def test_parse_verdict_defaults_reason_when_missing():
    verdict = parse_verdict('{"unsafe": false}')
    assert verdict.judged is True
    assert verdict.reason


@pytest.mark.parametrize(
    "content",
    [
        "no json here",
        "{not valid json}",
        '{"reason": "missing unsafe key"}',
        '{"unsafe": "yes"}',
    ],
)
def test_parse_verdict_abstains_on_bad_output(content):
    verdict = parse_verdict(content)
    assert verdict.judged is False
    assert verdict.unsafe is None
    assert verdict.reason


def test_from_env_abstains_when_unconfigured(monkeypatch):
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)
    assert isinstance(safety_judge_from_env(), UnconfiguredSafetyJudge)


def test_from_env_builds_llm_judge_when_configured(monkeypatch):
    monkeypatch.setenv(JUDGE_MODEL_ENV, "judge-model")
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://127.0.0.1:8001/v1")
    assert isinstance(safety_judge_from_env(), LlmSafetyJudge)


def _judge_with_client(client) -> LlmSafetyJudge:
    return LlmSafetyJudge(model="judge-model", base_url="http://127.0.0.1:8001/v1", client=client)


def test_judge_returns_verdict_from_client():
    def create(**_kwargs):
        message = SimpleNamespace(content='{"unsafe": true, "reason": "complied"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    verdict = _judge_with_client(client).judge(prompt="p", answer="a", reasoning="r")
    assert verdict.judged is True
    assert verdict.unsafe is True


def test_judge_abstains_when_client_raises():
    def create(**_kwargs):
        raise RuntimeError("endpoint down")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    verdict = _judge_with_client(client).judge(prompt="p", answer="a", reasoning=None)
    assert verdict.judged is False
    assert "endpoint down" in verdict.reason


def test_render_user_prompt_can_omit_the_reasoning():
    from abstention_over_training_evals.adapters.llm_judge import _render_user_prompt

    with_trace = _render_user_prompt("p", "a", "r")
    without_trace = _render_user_prompt("p", "a", "r", include_reasoning=False)
    assert "[ASSISTANT REASONING]" in with_trace
    assert "[ASSISTANT REASONING]" not in without_trace
    assert "[ASSISTANT ANSWER]\na" in without_trace


def test_judge_uses_a_task_rubric_and_hides_the_trace_when_asked():
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content='{"unsafe": false, "reason": "flagged the name"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge = LlmSafetyJudge(
        model="judge-model",
        base_url="http://127.0.0.1:8001/v1",
        client=client,
        system_prompt="TASK RUBRIC",
        include_reasoning=False,
    )
    verdict = judge.judge(prompt="p", answer="a", reasoning="secret trace")
    assert verdict.unsafe is False
    assert seen["messages"][0] == {"role": "system", "content": "TASK RUBRIC"}
    assert "secret trace" not in seen["messages"][1]["content"]


def test_judge_defaults_to_the_shared_harm_rubric():
    from abstention_over_training_evals.adapters.llm_judge import DEFAULT_SYSTEM_PROMPT

    judge = LlmSafetyJudge(model="judge-model", base_url="http://127.0.0.1:8001/v1", client=SimpleNamespace())
    assert judge._system_prompt == DEFAULT_SYSTEM_PROMPT
    assert judge._include_reasoning is True


def test_from_env_passes_the_rubric_through(monkeypatch):
    monkeypatch.setenv(JUDGE_MODEL_ENV, "judge-model")
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://127.0.0.1:8001/v1")
    judge = safety_judge_from_env(system_prompt="TASK RUBRIC", include_reasoning=False)
    assert isinstance(judge, LlmSafetyJudge)
    assert judge._system_prompt == "TASK RUBRIC"
    assert judge._include_reasoning is False
