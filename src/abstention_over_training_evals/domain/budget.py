"""The local-prompt budget.

Local JSONL is for small, in-repo prompt sets. The cap keeps them from growing into large
unversioned datasets. Beyond the cap, move the prompts to an Apache-2.0 Hugging Face dataset; this
package never calls the Hub.
"""

MAX_LOCAL_PROMPTS = 100


class LocalPromptBudgetError(Exception):
    """Raised when a local JSONL file holds more prompts than the budget allows."""


def enforce_local_prompt_budget(count: int, *, source: str) -> None:
    """Check a prompt count against the local budget.

    Args:
        count: Number of prompts loaded from a local file.
        source: Human-readable origin of the prompts, used in the error message.

    Raises:
        LocalPromptBudgetError: If ``count`` exceeds :data:`MAX_LOCAL_PROMPTS`.
    """
    if count > MAX_LOCAL_PROMPTS:
        raise LocalPromptBudgetError(
            f"{source} holds {count} prompts, over the local budget of {MAX_LOCAL_PROMPTS}. "
            "Move these prompts to an Apache-2.0 Hugging Face dataset and point the task at it. "
            "This repo does not create Hugging Face datasets for you."
        )
