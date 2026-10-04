"""Shared classes for single-turn safety-elicitation task plugins.

New tasks subclass :class:`LocalJsonlSafetyTask`, set ``NAME`` and ``PROMPTS_FILENAME``, and expose
a ``register_tasks`` entrypoint. The base class loads the local JSONL store, puts provenance on each
row, adds provenance metadata, and scores with :class:`SafetyJudgeMetric`. See
:mod:`abstention_over_training_evals.plugins.unsafe_elicitation` for an example and ``README.md`` for
a template.

This module has no ``register_tasks``, so the loader imports it for the shared classes but registers
nothing from it.
"""

import os
from pathlib import Path
from typing import Any

from eval_framework.contract import ResponseType
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion
from eval_framework.tasks.base import BaseTask, Language

from abstention_over_training_evals.adapters import JsonlPromptStore
from abstention_over_training_evals.adapters.llm_judge import safety_judge_from_env
from abstention_over_training_evals.adapters.olmo_settings import OLMO3_THINK_MAX_TOKENS
from abstention_over_training_evals.domain import PromptItem, SafetyJudge

# Prompts live under <repo>/data/local. `__file__` is .../src/<pkg>/plugins/base.py, so the repo
# root is three parents up.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "local"

# Per-task override (e.g. for tests or CI): `AOTE_PROMPTS_PATH__<NAME>`. Each task reads its own file,
# so there is no global override that could silently redirect every task at once.
_PROMPTS_PATH_ENV_PREFIX = "AOTE_PROMPTS_PATH__"


def resolve_prompts_path(task_name: str, filename: str) -> Path:
    override = os.environ.get(f"{_PROMPTS_PATH_ENV_PREFIX}{task_name}")
    if override:
        return Path(override)
    return _DATA_DIR / filename


class SafetyJudgeMetric(BaseMetric[Completion]):
    """Scores a completion with a :class:`SafetyJudge`.

    The verdict considers both the visible answer and the reasoning trace. An abstaining judge
    yields ``None`` (excluded from the mean) so "not judged" never reads as "safe". When no judge
    is injected, the judge is resolved from the environment (see
    :func:`abstention_over_training_evals.adapters.llm_judge.safety_judge_from_env`).
    """

    NAME = "Unsafe Elicitation Rate"

    def __init__(self, judge: SafetyJudge | None = None) -> None:
        self._judge: SafetyJudge = judge if judge is not None else safety_judge_from_env()

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=False, error=response.error)]

        judgement = self._judge.judge(
            prompt=response.prompt,
            answer=response.completion,
            reasoning=response.raw_completion_reasoning,
        )
        value: float | None
        if not judgement.judged:
            value = None
        elif judgement.unsafe is None:
            # `SafetyJudgement` forbids this; guarding anyway so a malformed verdict is excluded
            # (value None) rather than silently scored as safe (0.0).
            value = None
        else:
            value = float(judgement.unsafe)
        return [
            MetricResult(
                metric_name=self.NAME,
                value=value,
                higher_is_better=False,
                llm_judge_response=judgement.reason or None,
            )
        ]


class LocalJsonlSafetyTask(BaseTask[str]):
    """Base class for single-turn safety-elicitation tasks backed by a local JSONL prompt file.

    Subclasses set ``NAME`` (the eval-framework task name) and ``PROMPTS_FILENAME`` (the file under
    ``data/local/``). Prompts are single user turns with no gold answer; unsafe behaviour is judged,
    not matched.
    """

    # Set by subclasses.
    PROMPTS_FILENAME: str = ""

    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [SafetyJudgeMetric]
    SUBJECTS = ["default"]
    REVISION_LOCKFILE = None
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        if not type(self).PROMPTS_FILENAME:
            raise ValueError(f"{type(self).__name__} must set PROMPTS_FILENAME")
        super().__init__(num_fewshot=num_fewshot)
        # Apply the recommended token cap unless the CLI overrides it.
        self.max_tokens = OLMO3_THINK_MAX_TOKENS
        self._prompts_path = resolve_prompts_path(self.NAME, type(self).PROMPTS_FILENAME)
        # Resolve the class-level placeholder to the file actually read so run metadata is accurate.
        self.DATASET_PATH = str(self._prompts_path)
        self._items_cache: list[PromptItem] | None = None

    def _items(self) -> list[PromptItem]:
        # Load once; reused by `_load_dataset` and `get_metadata`.
        if self._items_cache is None:
            self._items_cache = JsonlPromptStore(self._prompts_path).load()
        return self._items_cache

    def _load_dataset(self, subject: str) -> None:
        # Prompts come from the local JSONL store, not `datasets.load_dataset`. Each row carries its
        # provenance.
        items = self._items()
        if not items:
            # The framework asserts a non-empty split; give an actionable message instead.
            raise ValueError(
                f"{self._prompts_path} contains no prompts. Add single-turn prompts (with "
                f"provenance) before running {self.NAME}, or run the SafetyPipelineSmokeTest task "
                "to check the pipeline end to end."
            )
        rows = [
            {"id": item.id, "prompt": item.prompt, "provenance": item.provenance.model_dump(mode="json")}
            for item in items
        ]
        self.dataset = {self.SAMPLE_SPLIT: rows, self.FEWSHOT_SPLIT: rows}

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta = super().get_metadata()
        items = self._items()
        meta["dataset_format"] = "local-jsonl"
        meta["prompt_count"] = str(len(items))
        meta["prompt_author_types"] = sorted({item.provenance.author_type.value for item in items})
        meta["prompt_generator_models"] = sorted(
            {item.provenance.generator_model for item in items if item.provenance.generator_model}
        )
        meta["prompt_generator_licenses"] = sorted(
            {item.provenance.generator_license for item in items if item.provenance.generator_license}
        )
        return meta

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return str(item["prompt"])

    def _get_ground_truth(self, item: dict[str, Any]) -> None:
        # No gold answer: unsafe behaviour is judged, not matched.
        return None
