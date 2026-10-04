"""IO- and framework-agnostic core: prompt items, the local-prompt budget, and judging ports.

This package may only depend on ``pydantic`` and ``loguru``. The import-linter contract in
``pyproject.toml`` enforces that boundary.
"""

from abstention_over_training_evals.domain.budget import (
    MAX_LOCAL_PROMPTS,
    LocalPromptBudgetError,
    enforce_local_prompt_budget,
)
from abstention_over_training_evals.domain.judging import (
    SafetyJudge,
    SafetyJudgement,
    unjudged,
)
from abstention_over_training_evals.domain.models import PromptItem
from abstention_over_training_evals.domain.provenance import (
    PERMISSIVE_LICENSES,
    AuthorType,
    Provenance,
)

__all__ = [
    "MAX_LOCAL_PROMPTS",
    "PERMISSIVE_LICENSES",
    "AuthorType",
    "LocalPromptBudgetError",
    "PromptItem",
    "Provenance",
    "SafetyJudge",
    "SafetyJudgement",
    "enforce_local_prompt_budget",
    "unjudged",
]
