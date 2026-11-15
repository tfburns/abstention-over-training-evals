"""The ``SelfHarmWeirdChat`` task: self-harm prompts from WeirdChat that Olmo 3 32B Think fails on.

Each row is a single-turn user prompt from the MIT-licensed `Transluce/WeirdChat` dataset (behaviors
``cutting-instructions``, ``direct-kill-yourself``, ``suicide-jumping``, ``extreme-calorie-restriction``
and ``purging-advice``), kept because Olmo 3 32B Think gave an unsafe response to it in a screening
pass. Every row carries a ``dataset`` provenance pointing at its WeirdChat prompt id.
"""

from eval_framework.tasks.registry import Registry, register_task

from abstention_over_training_evals.plugins.base import LocalJsonlSafetyTask

__all__ = ["SelfHarmWeirdChat", "register_tasks"]


class SelfHarmWeirdChat(LocalJsonlSafetyTask):
    """Self-harm prompts from WeirdChat that elicited unsafe responses from Olmo 3 32B Think."""

    NAME = "SelfHarmWeirdChat"
    PROMPTS_FILENAME = "selfharm_weirdchat.jsonl"


def register_tasks(registry: Registry) -> None:
    """Register this module's tasks with an eval-framework registry."""
    register_task(SelfHarmWeirdChat, registry=registry)
