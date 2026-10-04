from abstention_over_training_evals.adapters import UnconfiguredSafetyJudge
from abstention_over_training_evals.domain import SafetyJudge


def test_unconfigured_judge_abstains():
    verdict = UnconfiguredSafetyJudge().judge("prompt", "answer", "reasoning")
    assert verdict.judged is False
    assert verdict.unsafe is None


def test_unconfigured_judge_satisfies_the_port():
    judge: SafetyJudge = UnconfiguredSafetyJudge()
    assert isinstance(judge, SafetyJudge)
