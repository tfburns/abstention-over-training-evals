"""The ``UnsafeElicitation`` task: single-turn prompts probing for unsafe behaviour.

Each row is one original user prompt, sent as a single turn with no few-shot context. There is no
gold answer; a domain :class:`SafetyJudge` decides whether the response is unsafe (configured from
the environment, see ``README.md``). Shared classes live in
:mod:`abstention_over_training_evals.plugins.base`; this module is the example to copy for new tasks.

eval-framework loads this module via ``--extra-tasks-dir`` and calls :func:`register_tasks`.
"""

from eval_framework.tasks.registry import Registry, register_task

from abstention_over_training_evals.plugins.base import (
    LocalJsonlSafetyTask,
    SafetyJudgeMetric,
)

__all__ = ["SafetyJudgeMetric", "UnsafeElicitation", "register_tasks"]


class UnsafeElicitation(LocalJsonlSafetyTask):
    """Single-turn safety-elicitation prompts for Olmo 3 checkpoints."""

    NAME = "UnsafeElicitation"
    PROMPTS_FILENAME = "unsafe_elicitation.jsonl"


def register_tasks(registry: Registry) -> None:
    """Register this module's tasks with an eval-framework registry."""
    register_task(UnsafeElicitation, registry=registry)
