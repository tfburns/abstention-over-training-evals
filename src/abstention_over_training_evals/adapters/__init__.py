"""Adapters: the JSONL prompt store, the checkpoint catalog, and the vLLM-backed client.

This is the only place disk layout, the Olmo checkpoint names, and the OpenAI-compatible client
live.
"""

from abstention_over_training_evals.adapters.checkpoints import (
    FINAL_CHECKPOINT,
    OLMO3_32B_CHECKPOINTS,
    OlmoCheckpoint,
)
from abstention_over_training_evals.adapters.jsonl_store import JsonlPromptStore
from abstention_over_training_evals.adapters.unconfigured_judge import UnconfiguredSafetyJudge

# `Olmo3Think32B` is intentionally not re-exported here: importing it pulls in `eval_framework`,
# so importing the `adapters` package stays framework-free. Reference it by its full module path
# (`abstention_over_training_evals.adapters.olmo_vllm.Olmo3Think32B`), which is also how the
# eval-framework CLI loads it via `--llm-name`.

__all__ = [
    "FINAL_CHECKPOINT",
    "OLMO3_32B_CHECKPOINTS",
    "JsonlPromptStore",
    "OlmoCheckpoint",
    "UnconfiguredSafetyJudge",
]
