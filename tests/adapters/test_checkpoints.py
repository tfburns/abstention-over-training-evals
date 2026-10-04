from abstention_over_training_evals.adapters import (
    FINAL_CHECKPOINT,
    OLMO3_32B_CHECKPOINTS,
)
from abstention_over_training_evals.adapters.checkpoints import TrainingStage


def test_final_checkpoint_is_the_think_model():
    assert FINAL_CHECKPOINT.model_id == "allenai/Olmo-3-32B-Think"
    assert FINAL_CHECKPOINT.stage is TrainingStage.RLVR


def test_catalog_leads_with_the_final_checkpoint():
    assert OLMO3_32B_CHECKPOINTS[0] == FINAL_CHECKPOINT


def test_catalog_covers_each_stage_once():
    stages = {c.stage for c in OLMO3_32B_CHECKPOINTS}
    assert stages == set(TrainingStage)


def test_catalog_model_ids_match_hub_repos():
    # Checked against the Hub: `allenai/Olmo-3-32B` 404s; the base weights are Olmo-3-1125-32B.
    assert [checkpoint.model_id for checkpoint in OLMO3_32B_CHECKPOINTS] == [
        "allenai/Olmo-3-32B-Think",
        "allenai/Olmo-3-32B-Think-DPO",
        "allenai/Olmo-3-32B-Think-SFT",
        "allenai/Olmo-3-1125-32B",
    ]


def test_at_revision_pins_an_intermediate_checkpoint():
    intermediate = FINAL_CHECKPOINT.at_revision("step-10000", note="intermediate RLVR checkpoint")
    assert intermediate.revision == "step-10000"
    assert intermediate.model_id == FINAL_CHECKPOINT.model_id
    assert intermediate.stage is FINAL_CHECKPOINT.stage
    assert intermediate.note == "intermediate RLVR checkpoint"


def test_at_revision_leaves_the_original_unchanged():
    FINAL_CHECKPOINT.at_revision("step-10000")
    assert FINAL_CHECKPOINT.revision == "main"


def test_at_revision_keeps_the_note_when_omitted():
    intermediate = FINAL_CHECKPOINT.at_revision("step-10000")
    assert intermediate.note == FINAL_CHECKPOINT.note
