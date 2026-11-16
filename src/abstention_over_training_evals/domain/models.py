"""The items an eval row carries into the framework: a free-text prompt, or a multiple-choice item."""

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from abstention_over_training_evals.domain.provenance import Provenance


class PromptItem(BaseModel):
    """A single-turn user prompt with a stable id and provenance.

    There is no ground truth; unsafe behaviour is judged from the model's completion rather than
    matched to a gold string. Provenance keeps the dataset's authorship and licensing auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    prompt: str
    provenance: Provenance
    category: str | None = None

    @field_validator("id", "prompt", "category")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value


class McqItem(BaseModel):
    """A multiple-choice item for log-likelihood evaluation of pre-trained (base) checkpoints.

    Base checkpoints are not instruction-tuned, so instead of judging free text a task shows the
    question with lettered options and scores the likelihood of each letter. ``gold`` is the correct
    option and ``distractors`` are the wrong ones; the task shuffles them into a fixed order per item.
    ``control`` marks an item about a real entity, whose gold answer is a true specific: controls
    check that a model cannot score well by always picking a "this does not exist" kind of option.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    gold: str
    distractors: tuple[str, ...]
    provenance: Provenance
    category: str | None = None
    control: bool = False

    @field_validator("id", "question", "gold", "category")
    @classmethod
    def _non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value

    @field_validator("distractors")
    @classmethod
    def _distractors_well_formed(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("needs at least one distractor")
        if any(not option.strip() for option in value):
            raise ValueError("distractors must not be empty or whitespace")
        if len(set(value)) != len(value):
            raise ValueError("distractors must be distinct")
        return value

    @model_validator(mode="after")
    def _gold_is_not_a_distractor(self) -> "McqItem":
        if self.gold in self.distractors:
            raise ValueError("gold must not also appear among the distractors")
        return self
