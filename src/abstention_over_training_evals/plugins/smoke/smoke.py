"""A benign smoke-test task for the serve/eval pipeline.

It carries one harmless prompt so you can confirm serving, plugin loading, generation, and judging
work before writing elicitation prompts. It is separate from ``UnsafeElicitation`` so the
elicitation dataset holds only elicitation prompts.

eval-framework loads this module via ``--extra-tasks-dir`` and calls :func:`register_tasks`.
"""

from eval_framework.tasks.registry import Registry, register_task

from abstention_over_training_evals.plugins.base import LocalJsonlSafetyTask

__all__ = ["SafetyPipelineSmokeTest", "register_tasks"]


class SafetyPipelineSmokeTest(LocalJsonlSafetyTask):
    """A single benign prompt for smoke-testing the pipeline."""

    NAME = "SafetyPipelineSmokeTest"
    PROMPTS_FILENAME = "smoke.jsonl"


def register_tasks(registry: Registry) -> None:
    """Register this module's tasks with an eval-framework registry."""
    register_task(SafetyPipelineSmokeTest, registry=registry)
