"""The Olmo 3 32B checkpoint catalog.

Names only: this repo does not download weights. A local ``vllm serve`` points at a checkpoint and
resolves the Hub reference itself. The first target is the final post-trained checkpoint; the
earlier post-training stages and the pre-trained base are listed so later work can evaluate them and
compare their behaviour with the final checkpoint.

The ``model_id`` values are the expected Hugging Face references. This package makes no Hub calls, so
a wrong name fails only when a human runs ``vllm serve``; check each id against the Hub first.

Intermediate post-training checkpoints are Hub revisions (branches) of each stage repo. Reference
one with :meth:`OlmoCheckpoint.at_revision`; the catalog lists each stage's final (``main``)
checkpoint.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TrainingStage(StrEnum):
    """Stage in the Olmo 3 32B Think training flow."""

    BASE = "base"
    SFT = "sft"
    DPO = "dpo"
    RLVR = "rlvr"


class OlmoCheckpoint(BaseModel):
    """A single Hugging Face model reference and its stage in the flow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    stage: TrainingStage
    # For post-training, the Hub exposes intermediate checkpoints as revisions (branches). The
    # default ``main`` is the stage's final checkpoint.
    revision: str = "main"
    note: str = ""

    def at_revision(self, revision: str, note: str = "") -> OlmoCheckpoint:
        """Return a copy of this checkpoint pinned to an intermediate Hub revision (branch).

        Post-training exposes intermediate checkpoints as revisions of the stage repo; serve one
        with ``vllm serve <model_id> --revision <revision>``. Revision names must be verified
        against the Hub (this package makes no Hub calls).

        Args:
            revision: The Hub revision (branch or commit) to pin to.
            note: Optional note; defaults to this checkpoint's note when omitted.

        Returns:
            A new frozen :class:`OlmoCheckpoint`; the original is unchanged.
        """
        return self.model_copy(update={"revision": revision, "note": note or self.note})


# The final post-trained checkpoint (RLVR on top of DPO); the first evaluation target.
FINAL_CHECKPOINT = OlmoCheckpoint(
    model_id="allenai/Olmo-3-32B-Think",
    stage=TrainingStage.RLVR,
    note="Final post-trained Olmo 3 32B Think model; primary evaluation target.",
)


OLMO3_32B_CHECKPOINTS: tuple[OlmoCheckpoint, ...] = (
    FINAL_CHECKPOINT,
    OlmoCheckpoint(
        model_id="allenai/Olmo-3-32B-Think-DPO",
        stage=TrainingStage.DPO,
        note="Post-training stage 2: DPO.",
    ),
    OlmoCheckpoint(
        model_id="allenai/Olmo-3-32B-Think-SFT",
        stage=TrainingStage.SFT,
        note="Post-training stage 1: SFT.",
    ),
    OlmoCheckpoint(
        # The model card's "Olmo-3-32B" label points at this repo. `allenai/Olmo-3-32B` is not a
        # Hub repository.
        model_id="allenai/Olmo-3-1125-32B",
        stage=TrainingStage.BASE,
        note="Pre-trained base model (Hub repo Olmo-3-1125-32B).",
    ),
)
