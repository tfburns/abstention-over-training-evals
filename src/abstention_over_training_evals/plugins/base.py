"""Shared classes for single-turn safety-elicitation task plugins and their base-model variants.

New tasks subclass :class:`LocalJsonlSafetyTask`, set ``NAME`` and ``PROMPTS_FILENAME``, and expose
a ``register_tasks`` entrypoint. The base class loads the local JSONL store, puts provenance on each
row, adds provenance metadata, and scores with :class:`SafetyJudgeMetric`. See
:mod:`abstention_over_training_evals.plugins.unsafe_elicitation` for an example and ``README.md`` for
a template.

Pre-trained (base) checkpoints are not instruction-tuned and cannot be judged on free text, so a
task's base-model variant subclasses :class:`LocalJsonlMcqTask` instead: the same question as a
multiple-choice item scored by log-likelihood over the lettered options (see
:mod:`abstention_over_training_evals.plugins.nonexistent_entities_mcq` for an example).

This module has no ``register_tasks``, so the loader imports it for the shared classes but registers
nothing from it.
"""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval_framework.contract import ResponseType, Sample
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion
from eval_framework.tasks.base import BaseTask, Language
from eval_framework.tasks.task_style import IdkStyle, MCStyle, shuffle_correct_with_distractors

from abstention_over_training_evals.adapters import JsonlMcqStore, JsonlPromptStore
from abstention_over_training_evals.adapters.llm_judge import safety_judge_from_env
from abstention_over_training_evals.adapters.olmo_settings import OLMO3_THINK_MAX_TOKENS
from abstention_over_training_evals.domain import McqItem, PromptItem, SafetyJudge

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
            {
                "id": item.id,
                "prompt": item.prompt,
                "provenance": item.provenance.model_dump(mode="json"),
                "category": item.category,
            }
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
        meta["prompt_categories"] = sorted({item.category for item in items if item.category})
        return meta

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return str(item["prompt"])

    def _get_ground_truth(self, item: dict[str, Any]) -> None:
        # No gold answer: unsafe behaviour is judged, not matched.
        return None


class MCStyleWithShownAbstention(MCStyle):
    """``MCStyle`` whose last lettered option is an abstention the model can see.

    eval-framework's ``IdkStyle`` scores a hidden abstention completion next to the lettered options.
    A base model has no way to learn what a hidden completion means, so here the abstention is a
    visible option (``E. I don't know``) chosen by its letter like any other. The confidence-aware
    metrics (Distributional Correctness Score, Ternary Score) treat the last scored completion as the
    abstention, which is why :class:`LocalJsonlMcqTask` appends it after the shuffled real options.
    """

    metrics: list[type[BaseMetric[Any]]] = list(IdkStyle.metrics)


class LocalJsonlMcqTask(BaseTask[str]):
    """Base class for log-likelihood multiple-choice tasks backed by a local JSONL item file.

    The base-model analogue of :class:`LocalJsonlSafetyTask`. Subclasses set ``NAME``,
    ``PROMPTS_FILENAME`` (the file under ``data/local/``) and ``TASK_STYLER`` (``MCStyle()`` for a
    plain multiple-choice task, :class:`MCStyleWithShownAbstention` with ``ABSTENTION_OPTION`` set
    to also offer an abstention). Each item's gold answer and distractors are shuffled into a fixed
    order (seeded by the item id and question, so the plain and abstention variants of a task show
    the same order) and the model is scored on the letters. Items are reported under two subjects:
    ``main`` and, when the file has any, ``control`` (items about real entities).
    """

    # Set by subclasses.
    PROMPTS_FILENAME: str = ""
    # When set, shown as the last lettered option of every item (the model can abstain by letter).
    ABSTENTION_OPTION: str | None = None

    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    SUBJECTS = ["main", "control"]
    REVISION_LOCKFILE = None
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        if not type(self).PROMPTS_FILENAME:
            raise ValueError(f"{type(self).__name__} must set PROMPTS_FILENAME")
        super().__init__(num_fewshot=num_fewshot)
        self._prompts_path = resolve_prompts_path(self.NAME, type(self).PROMPTS_FILENAME)
        self.DATASET_PATH = str(self._prompts_path)
        self._items_cache: list[McqItem] | None = None

    def _items(self) -> list[McqItem]:
        if self._items_cache is None:
            self._items_cache = JsonlMcqStore(self._prompts_path).load()
        return self._items_cache

    @staticmethod
    def _subject_of(item: McqItem) -> str:
        return "control" if item.control else "main"

    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        # Only subjects that have items: the framework asserts a non-empty split per subject.
        items = self._items()
        if not items:
            raise ValueError(
                f"{self._prompts_path} contains no items. Add multiple-choice items (question, gold, "
                f"distractors, provenance) before running {self.NAME}."
            )
        present = {self._subject_of(item) for item in items}
        self.SUBJECTS = [subject for subject in self.SUBJECTS if subject in present]
        yield from super().iterate_samples(num_samples)

    def _load_dataset(self, subject: str) -> None:
        rows = []
        for item in self._items():
            if self._subject_of(item) != subject:
                continue
            choices, correct_index = shuffle_correct_with_distractors(
                item.gold, list(item.distractors), seed_text=f"{item.id}\n{item.question}"
            )
            abstention = type(self).ABSTENTION_OPTION
            if abstention is not None:
                choices = [*choices, abstention]
            rows.append(
                {
                    "id": item.id,
                    "category": item.category,
                    "control": item.control,
                    "question": item.question,
                    "choices": choices,
                    "correct_index": correct_index,
                    "provenance": item.provenance.model_dump(mode="json"),
                }
            )
        self.dataset = {self.SAMPLE_SPLIT: rows, self.FEWSHOT_SPLIT: rows}

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta = super().get_metadata()
        items = self._items()
        meta["dataset_format"] = "local-jsonl-mcq"
        meta["prompt_count"] = str(len(items))
        meta["control_count"] = str(sum(1 for item in items if item.control))
        meta["prompt_author_types"] = sorted({item.provenance.author_type.value for item in items})
        meta["prompt_categories"] = sorted({item.category for item in items if item.category})
        meta["abstention_option"] = type(self).ABSTENTION_OPTION or ""
        return meta

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return str(item["question"])

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return list(item["choices"])

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return int(item["correct_index"])
