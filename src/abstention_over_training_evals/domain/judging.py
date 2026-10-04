"""The safety-judging port.

A judge decides whether a model's response is unsafe. The port lives in the domain so the task and
its metric can depend on a judge without binding to a backend; the default adapter abstains.
"""

from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, model_validator


class SafetyJudgement(BaseModel):
    """The verdict on a single response.

    ``judged`` is ``False`` when no judge was configured, which keeps "we did not look" distinct
    from "we looked and it was safe". A judged verdict sets ``unsafe``; an abstaining one leaves it
    ``None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    judged: bool
    unsafe: bool | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _check_verdict(self) -> Self:
        if self.judged and self.unsafe is None:
            raise ValueError("a judged verdict must set `unsafe`")
        if not self.judged and self.unsafe is not None:
            raise ValueError("an abstaining verdict must leave `unsafe` as None")
        return self


@runtime_checkable
class SafetyJudge(Protocol):
    """Scores whether a response elicited unsafe behaviour.

    Implementations must consider the answer and the reasoning trace, because a Think model can put
    unsafe content in the ``<think>`` block even when the answer looks clean.
    """

    def judge(self, prompt: str, answer: str, reasoning: str | None) -> SafetyJudgement:
        """Return a verdict for one prompt/response pair."""
        ...


def unjudged(reason: str = "no safety judge configured") -> SafetyJudgement:
    """Return the abstaining verdict (used when no judge is configured)."""
    return SafetyJudgement(judged=False, unsafe=None, reason=reason)
