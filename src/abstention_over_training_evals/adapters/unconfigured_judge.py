"""The default safety judge: always abstains.

It implements the domain :class:`SafetyJudge` port and reports "not judged", so generation can run
before a judge is configured.
"""

from abstention_over_training_evals.domain import SafetyJudgement, unjudged


class UnconfiguredSafetyJudge:
    """A :class:`SafetyJudge` that never returns a verdict."""

    def judge(self, prompt: str, answer: str, reasoning: str | None) -> SafetyJudgement:
        return unjudged()
