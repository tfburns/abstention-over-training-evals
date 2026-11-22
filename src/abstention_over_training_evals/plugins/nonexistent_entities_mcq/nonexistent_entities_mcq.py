"""Multiple-choice variants of the NonexistentEntities tasks, for pre-trained (base) checkpoints.

The generative ``NonexistentEntities`` task asks about entities that sound real but are not and judges
whether the final answer presents specifics about them as real. A base checkpoint is not
instruction-tuned and will not answer such a question, so this plugin asks the same questions as
multiple-choice items scored by log-likelihood: the lettered options are the author's "this does not
exist" answer (gold) and three wrong options that treat the entity as real (a fabricated specific, a
second fabricated specific, a silent substitution by a real neighbour), in a fixed per-item shuffle.
Five real-entity controls, whose gold answer is a true specific, check that a model cannot score well
by always choosing the "does not exist" option.

Four tasks read the same item file ``data/local/nonexistent_entities_mcq.jsonl``, in the two
formulations OLMES (Gu et al., 2024, arXiv:2406.08446, section 2.1) recommends evaluating side by side:

- ``NonexistentEntitiesMCQ``: multiple-choice formulation, the four options shown as lettered lines and
  the letters scored.
- ``NonexistentEntitiesMCQIdk``: the same with a fifth, visible option ``E. I don't know``, so a model
  can abstain by letter; scored with the confidence-aware metrics as well (Distributional Correctness
  Score, Ternary Score), which treat that last option as the abstention.
- ``NonexistentEntitiesMCQCloze``: cloze formulation, the question alone shown and each option's text
  scored. Option length matters here, so the Bayesian (length-corrected) accuracy is the one to read.
- ``NonexistentEntitiesMCQClozeIdk``: the cloze formulation with a hidden ``I don't know`` completion
  scored after the options, eval-framework's ``IdkStyle``.

Items are written by hand in a curation sheet and converted with
``scripts/nonexistent_entities_mcq/sheet_to_jsonl.py``; see the plugin README.
"""

from eval_framework.tasks.registry import Registry, register_task
from eval_framework.tasks.task_style import ClozeStyle, MCStyle

from abstention_over_training_evals.plugins.base import LocalJsonlMcqTask, MCStyleWithShownAbstention

IDK_OPTION = "I don't know"

_ITEMS_FILENAME = "nonexistent_entities_mcq.jsonl"


class NonexistentEntitiesMCQ(LocalJsonlMcqTask):
    """Fake-entity questions as four-option multiple choice.

    The gold option says the entity does not exist; for the real-entity controls it is the true specific.
    """

    NAME = "NonexistentEntitiesMCQ"
    PROMPTS_FILENAME = _ITEMS_FILENAME
    TASK_STYLER = MCStyle()


class NonexistentEntitiesMCQIdk(LocalJsonlMcqTask):
    """The same items with a visible fifth option, ``E. I don't know``, and confidence-aware scoring."""

    NAME = "NonexistentEntitiesMCQIdk"
    PROMPTS_FILENAME = _ITEMS_FILENAME
    TASK_STYLER = MCStyleWithShownAbstention()
    ABSTENTION_OPTION = IDK_OPTION


class NonexistentEntitiesMCQCloze(LocalJsonlMcqTask):
    """The same items in cloze form: only the question is shown and each option's text is scored."""

    NAME = "NonexistentEntitiesMCQCloze"
    PROMPTS_FILENAME = _ITEMS_FILENAME
    TASK_STYLER = ClozeStyle()


class NonexistentEntitiesMCQClozeIdk(LocalJsonlMcqTask):
    """Cloze form with a hidden ``I don't know`` completion scored after the options."""

    NAME = "NonexistentEntitiesMCQClozeIdk"
    PROMPTS_FILENAME = _ITEMS_FILENAME
    TASK_STYLER = ClozeStyle().with_abstention_option(f" {IDK_OPTION}")
    SCORED_ABSTENTION = IDK_OPTION


def register_tasks(registry: Registry) -> None:
    """Register this module's tasks with an eval-framework registry."""
    register_task(NonexistentEntitiesMCQ, registry=registry)
    register_task(NonexistentEntitiesMCQIdk, registry=registry)
    register_task(NonexistentEntitiesMCQCloze, registry=registry)
    register_task(NonexistentEntitiesMCQClozeIdk, registry=registry)
