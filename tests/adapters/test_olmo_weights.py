import pytest

from abstention_over_training_evals.adapters.checkpoints import OLMO3_32B_CHECKPOINTS, TrainingStage
from abstention_over_training_evals.adapters.olmo_weights import (
    WEIGHT_FILE_PATTERNS,
    WEIGHTS_DIR_ENV,
    checkpoint_label,
    download_weights,
    parse_checkpoint,
    remove_weights,
    weights_dir,
    weights_root,
)


@pytest.mark.parametrize(
    ("spec", "model_id", "stage", "revision"),
    [
        ("base", "allenai/Olmo-3-1125-32B", TrainingStage.BASE, "main"),
        ("sft", "allenai/Olmo-3-32B-Think-SFT", TrainingStage.SFT, "main"),
        ("dpo", "allenai/Olmo-3-32B-Think-DPO", TrainingStage.DPO, "main"),
        ("think", "allenai/Olmo-3-32B-Think", TrainingStage.RLVR, "main"),
        ("rlvr", "allenai/Olmo-3-32B-Think", TrainingStage.RLVR, "main"),
        ("base@stage1-step72000", "allenai/Olmo-3-1125-32B", TrainingStage.BASE, "stage1-step72000"),
        ("think@step_100", "allenai/Olmo-3-32B-Think", TrainingStage.RLVR, "step_100"),
        ("allenai/Olmo-3-32B-Think-SFT@abc123", "allenai/Olmo-3-32B-Think-SFT", TrainingStage.SFT, "abc123"),
        (" SFT ", "allenai/Olmo-3-32B-Think-SFT", TrainingStage.SFT, "main"),
    ],
)
def test_parse_checkpoint_resolves_aliases_ids_and_revisions(spec, model_id, stage, revision):
    checkpoint = parse_checkpoint(spec)
    assert checkpoint.model_id == model_id
    assert checkpoint.stage is stage
    assert checkpoint.revision == revision


def test_parse_checkpoint_keeps_the_catalog_note():
    base = next(c for c in OLMO3_32B_CHECKPOINTS if c.stage is TrainingStage.BASE)
    assert parse_checkpoint("base@stage1-step1000").note == base.note


def test_parse_checkpoint_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown checkpoint 'allenai/Olmo-2-7B'"):
        parse_checkpoint("allenai/Olmo-2-7B@main")


def test_checkpoint_label_is_filesystem_safe():
    assert checkpoint_label(parse_checkpoint("base@stage1-step72000")) == "allenai_Olmo-3-1125-32B@stage1-step72000"
    assert checkpoint_label(parse_checkpoint("think")) == "allenai_Olmo-3-32B-Think@main"


def test_weights_root_prefers_argument_then_env_then_default(tmp_path, monkeypatch):
    monkeypatch.delenv(WEIGHTS_DIR_ENV, raising=False)
    assert weights_root().name == "aote-weights"
    monkeypatch.setenv(WEIGHTS_DIR_ENV, str(tmp_path / "env"))
    assert weights_root() == tmp_path / "env"
    assert weights_root(tmp_path / "arg") == tmp_path / "arg"
    expected = tmp_path / "arg" / "allenai_Olmo-3-32B-Think-SFT@main"
    assert weights_dir(parse_checkpoint("sft"), tmp_path / "arg") == expected


def test_download_weights_fetches_only_the_loadable_files_into_the_weights_dir(tmp_path, monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    target = download_weights(parse_checkpoint("base@stage1-step72000"), tmp_path)
    assert target == tmp_path / "allenai_Olmo-3-1125-32B@stage1-step72000"
    assert calls == [
        {
            "repo_id": "allenai/Olmo-3-1125-32B",
            "revision": "stage1-step72000",
            "local_dir": str(target),
            "allow_patterns": list(WEIGHT_FILE_PATTERNS),
        }
    ]
    assert "*.safetensors" in WEIGHT_FILE_PATTERNS
    assert "*.json" in WEIGHT_FILE_PATTERNS


def test_remove_weights_deletes_the_checkpoint_dir_and_tolerates_absence(tmp_path):
    checkpoint = parse_checkpoint("dpo")
    target = weights_dir(checkpoint, tmp_path)
    target.mkdir(parents=True)
    (target / "model.safetensors").write_bytes(b"x")
    remove_weights(checkpoint, tmp_path)
    assert not target.exists()
    remove_weights(checkpoint, tmp_path)  # no error the second time
