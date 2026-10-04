"""The prompt item: the single unit an eval row carries into the framework."""

from pydantic import BaseModel, ConfigDict, field_validator

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

    @field_validator("id", "prompt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty or whitespace")
        return value
