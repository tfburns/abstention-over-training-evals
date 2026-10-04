import pytest
from pydantic import ValidationError

from abstention_over_training_evals.domain import SafetyJudge, SafetyJudgement, unjudged


def test_unjudged_abstains():
    verdict = unjudged()
    assert verdict.judged is False
    assert verdict.unsafe is None
    assert verdict.reason


def test_safety_judgement_accepts_a_committed_verdict():
    verdict = SafetyJudgement(judged=True, unsafe=True, reason="because")
    assert verdict.unsafe is True


def test_judged_verdict_must_set_unsafe():
    with pytest.raises(ValidationError):
        SafetyJudgement(judged=True, reason="forgot to decide")


def test_abstaining_verdict_must_not_set_unsafe():
    with pytest.raises(ValidationError):
        SafetyJudgement(judged=False, unsafe=False)


class _FakeJudge:
    def judge(self, prompt: str, answer: str, reasoning: str | None) -> SafetyJudgement:
        return SafetyJudgement(judged=True, unsafe=False, reason="ok")


def test_protocol_is_satisfied_structurally():
    judge: SafetyJudge = _FakeJudge()
    assert isinstance(judge, SafetyJudge)
    assert judge.judge("p", "a", None).unsafe is False
