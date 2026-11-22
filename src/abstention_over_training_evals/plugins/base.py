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

import itertools
import os
from collections.abc import Iterable, Sequence
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

    # A task whose failure mode is not operational harm overrides these on a metric subclass: the
    # rubric text the judge is given (``None`` keeps the default harm rubric) and whether the judge
    # also sees the reasoning trace. eval-framework builds metrics with no arguments, so class
    # attributes are the only per-task hook.
    JUDGE_SYSTEM_PROMPT: str | None = None
    JUDGE_INCLUDES_REASONING: bool = True

    def __init__(self, judge: SafetyJudge | None = None) -> None:
        cls = type(self)
        self._judge: SafetyJudge = (
            judge
            if judge is not None
            else safety_judge_from_env(
                system_prompt=cls.JUDGE_SYSTEM_PROMPT, include_reasoning=cls.JUDGE_INCLUDES_REASONING
            )
        )

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
    visible option (``I don't know``) chosen by its letter like any other. The confidence-aware
    metrics (Distributional Correctness Score, Ternary Score) treat the last scored completion as the
    abstention, which is why :class:`LocalJsonlMcqTask` scores the abstention's letter last whichever
    letter it is shown under.
    """

    metrics: list[type[BaseMetric[Any]]] = list(IdkStyle.metrics)


class LocalJsonlMcqTask(BaseTask[str]):
    """Base class for log-likelihood multiple-choice tasks backed by a local JSONL item file.

    The base-model analogue of :class:`LocalJsonlSafetyTask`. Subclasses set ``NAME``,
    ``PROMPTS_FILENAME`` (the file under ``data/local/``) and ``TASK_STYLER``:

    - ``MCStyle()`` shows the options as lettered lines and scores the letters;
      :class:`MCStyleWithShownAbstention` with ``ABSTENTION_OPTION`` set also offers a visible
      abstention as the last letter.
    - ``ClozeStyle()`` shows the question alone and scores each option's text;
      ``ClozeStyle().with_abstention_option(" ...")`` with ``SCORED_ABSTENTION`` set also scores a
      hidden abstention completion after the options (eval-framework's ``IdkStyle``).

    Each item's gold answer and distractors are shuffled into a fixed base order (seeded by the item id
    and question, so every variant of a task scores the same order); a visible abstention goes after
    them. ``OPTION_ORDERS`` then says how the shown options are ordered across samples: ``"fixed"``
    (the base order only), ``"cyclic"`` (every cyclic rotation, one sample each) or ``"all"`` (every
    permutation, one sample each; 24 for four options, 120 for five). Under ``"cyclic"`` and ``"all"``
    every option, the abstention included, is shown under every letter equally often, so the mean over
    samples carries no letter-position bias; ``"all"`` also varies the neighbours. Order 0 is always
    the base order. The cloze tasks show no options, so they stay ``"fixed"``. Items are reported under
    two subjects: ``main`` and, when the file has any, ``control`` (items about real entities).
    """

    # Set by subclasses.
    PROMPTS_FILENAME: str = ""
    # When set, shown as the last lettered option of every item (the model can abstain by letter).
    ABSTENTION_OPTION: str | None = None
    # When set, the abstention text that ``TASK_STYLER`` scores as a hidden extra completion
    # (cloze tasks); recorded in the metadata only, the styler adds the completion itself.
    SCORED_ABSTENTION: str | None = None
    # How the shown options are ordered across an item's samples: "fixed", "cyclic" or "all" (see the
    # class docstring). Only meaningful for tasks that show the options.
    OPTION_ORDERS: str = "fixed"

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

    def _present_subjects(self) -> list[str]:
        # Only subjects that have items: the framework asserts a non-empty split per subject.
        items = self._items()
        if not items:
            raise ValueError(
                f"{self._prompts_path} contains no items. Add multiple-choice items (question, gold, "
                f"distractors, provenance) before running {self.NAME}."
            )
        present = {self._subject_of(item) for item in items}
        return [subject for subject in self.SUBJECTS if subject in present]

    def iterate_samples(self, num_samples: int | None = None) -> Iterable[Sample]:
        self.SUBJECTS = self._present_subjects()
        if num_samples is not None:
            # Keep whole items: a cap that stops mid-item would break the letter balance for that item.
            per_item = len(
                self.option_orders(
                    max(len(item.distractors) for item in self._items())
                    + 1
                    + (type(self).ABSTENTION_OPTION is not None)
                )
            )
            num_samples = -(-num_samples // per_item) * per_item
        yield from super().iterate_samples(num_samples)

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        # The framework excludes only an example equal to the item; a reordered copy of the item is
        # not equal to its base-order row, so exclude by item id instead.
        assert self.rnd is not None
        pool = [example for example in self.dataset[self.FEWSHOT_SPLIT] if example["id"] != item["id"]]
        # A subject with fewer other items than requested (e.g. a lone control) gets what there is.
        return self.rnd.sample(pool, min(self.num_fewshot, len(pool)))

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
                    # Position of the visible abstention in ``choices`` (None without one).
                    "abstention_index": len(choices) - 1 if abstention is not None else None,
                    "order": 0,
                    "provenance": item.provenance.model_dump(mode="json"),
                }
            )
        self.dataset = {self.SAMPLE_SPLIT: rows, self.FEWSHOT_SPLIT: rows}

    @classmethod
    def option_orders(cls, n: int) -> list[tuple[int, ...]]:
        """The orders ``OPTION_ORDERS`` prescribes for ``n`` options, order 0 being the base order.

        Each order lists, per shown position, the index of the base-order option shown there.
        """
        mode = cls.OPTION_ORDERS
        if mode == "fixed":
            return [tuple(range(n))]
        if mode == "cyclic":
            return [tuple((j + k) % n for j in range(n)) for k in range(n)]
        if mode == "all":
            return list(itertools.permutations(range(n)))  # lexicographic, so the identity comes first
        raise ValueError(f"{cls.__name__}.OPTION_ORDERS must be 'fixed', 'cyclic' or 'all', not {mode!r}")

    @staticmethod
    def reorder_options(item: dict[str, Any], order: Sequence[int], index: int) -> dict[str, Any]:
        """Return a copy of ``item`` showing base-order option ``order[j]`` at position ``j``.

        ``correct_index`` and ``abstention_index`` follow their options; ``order`` records ``index``.
        """
        reordered = dict(item, choices=[item["choices"][i] for i in order], order=index)
        reordered["correct_index"] = list(order).index(int(item["correct_index"]))
        if item.get("abstention_index") is not None:
            reordered["abstention_index"] = list(order).index(int(item["abstention_index"]))
        return reordered

    def _orders(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        orders = self.option_orders(len(item["choices"]))
        if len(orders) == 1:
            return [item]
        return [self.reorder_options(item, order, index) for index, order in enumerate(orders)]

    def _create_samples(self, item: dict[str, Any], index: int, subject: str) -> list[Sample]:
        samples: list[Sample] = []
        for offset, reordered in enumerate(self._orders(item)):
            samples.extend(super()._create_samples(reordered, index + offset, subject))
        return samples

    def sample_manifest(self) -> list[dict[str, Any]]:
        """One row per sample, in ``iterate_samples`` order, saying what the sample scores.

        A run's ``output.jsonl`` identifies a sample only by ``subject`` and ``id``. This maps them back
        to the item, the option order, the shown options, the gold completion and the abstention
        completion (``None`` without one), for the per-item analysis in
        ``scripts/nonexistent_entities_mcq/summarise_runs.py``.
        """
        has_abstention = type(self).ABSTENTION_OPTION is not None or type(self).SCORED_ABSTENTION is not None
        rows: list[dict[str, Any]] = []
        for subject in self._present_subjects():
            self._load_dataset(subject)
            index = 0
            for item in self.dataset[self.SAMPLE_SPLIT]:
                for reordered in self._orders(item):
                    completions = list(self._get_possible_completions(reordered) or [])
                    rows.append(
                        {
                            "subject": subject,
                            "id": index,
                            "item_id": reordered["id"],
                            "category": reordered["category"],
                            "control": reordered["control"],
                            "question": reordered["question"],
                            "order": reordered["order"],
                            "choices": list(reordered["choices"]),
                            "completions": completions,
                            "gold": self._get_ground_truth(reordered),
                            "abstention": completions[-1] if has_abstention and completions else None,
                        }
                    )
                    index += 1
        return rows

    def get_metadata(self) -> dict[str, str | list[str]]:
        meta = super().get_metadata()
        items = self._items()
        meta["dataset_format"] = "local-jsonl-mcq"
        meta["prompt_count"] = str(len(items))
        meta["control_count"] = str(sum(1 for item in items if item.control))
        meta["prompt_author_types"] = sorted({item.provenance.author_type.value for item in items})
        meta["prompt_categories"] = sorted({item.category for item in items if item.category})
        meta["abstention_option"] = type(self).ABSTENTION_OPTION or type(self).SCORED_ABSTENTION or ""
        meta["option_orders"] = type(self).OPTION_ORDERS
        meta["sample_count"] = str(len(self.sample_manifest()))
        return meta

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return str(item["question"])

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return list(item["choices"])

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return int(item["correct_index"])

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        abstention_index = item.get("abstention_index")
        if completions is None or abstention_index is None:
            return completions
        # The confidence-aware metrics take the last scored completion as the abstention, so its
        # letter is scored last whichever letter the prompt shows it under.
        abstention_index = int(abstention_index)
        return [c for i, c in enumerate(completions) if i != abstention_index] + [completions[abstention_index]]
