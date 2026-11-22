"""The summariser's statistics, on a synthetic run built from a task's own manifest."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "nonexistent_entities_mcq" / "summarise_runs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("summarise_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _item(id, question, gold, distractors, control=False):
    return json.dumps(
        {
            "id": id,
            "category": "prescription_drug",
            "control": control,
            "question": question,
            "gold": gold,
            "distractors": distractors,
            "provenance": {"author_type": "human"},
        }
    )


@pytest.fixture
def items_file(tmp_path, monkeypatch):
    f = tmp_path / "items.jsonl"
    f.write_text(
        "\n".join(
            [
                _item("pharma-01", "What dose of hetinopril?", "Hetinopril does not exist", ["10 mg", "5 mg", "20 mg"]),
                _item("pharma-02", "What dose of cecafran?", "Cecafran does not exist", ["1 mg", "2 mg", "3 mg"]),
                _item(
                    "ctrl-01", "How is lamotrigine started?", "At 25 mg daily", ["No such drug", "500 mg", "1 g"], True
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AOTE_PROMPTS_PATH__NonexistentEntitiesMCQIdk", str(f))
    return f


def _responses(manifest, choose):
    """Fake output.jsonl rows: ``choose(row)`` names the completion to give the highest log-likelihood."""
    rows = []
    for row in manifest.values():
        winner = choose(row)
        lls = {c: (-1.0 if c == winner else -5.0) for c in row["completions"]}
        rows.append(
            {
                "id": row["id"],
                "subject": row["subject"],
                "prompt": f"Question: {row['question']}\n...",
                "loglikelihoods": lls,
            }
        )
    return rows


def test_always_picking_letter_a_shows_as_position_bias(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    assert len(manifest) == 3 * 120
    stats = script.summarise_run("NonexistentEntitiesMCQIdk", _responses(manifest, lambda row: " A"), [], manifest)
    main = stats["main"]
    assert main["n_items"] == 2
    assert main["n_samples"] == 240
    # Letter A holds each option equally often: the gold a fifth of the time, the abstention likewise.
    assert main["accuracy_all_orders"] == pytest.approx(0.2)
    assert main["abstention_rate"] == pytest.approx(0.2)
    assert main["consistency"] == 0.0  # the content behind A changes with the order
    assert main["letter_preference"] == {"A": 1.0}
    assert stats["control"]["accuracy_all_orders"] == pytest.approx(0.2)


def test_always_picking_the_gold_is_consistent_and_order_free(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    stats = script.summarise_run(
        "NonexistentEntitiesMCQIdk", _responses(manifest, lambda row: row["gold"]), [], manifest
    )
    import math

    e1, e5 = math.exp(-1.0), math.exp(-5.0)
    p_c = e1 / (e1 + 4 * e5)
    p_idk = e5 / (e1 + 4 * e5)
    p_w = 1 - p_c - p_idk
    for subject in ("main", "control"):
        assert stats[subject]["accuracy_all_orders"] == 1.0
        assert stats[subject]["accuracy_order_0"] == 1.0
        assert stats[subject]["abstention_rate"] == 0.0
        assert stats[subject]["consistency"] == 1.0
        assert stats[subject]["p_gold"] == pytest.approx(p_c)
        assert stats[subject]["dcs_raw"] == pytest.approx((p_c - p_w) * (1 - p_idk))
    # Across an item's 120 orders the gold sits under every letter equally often.
    assert stats["main"]["letter_preference"] == {k: pytest.approx(0.2) for k in "ABCDE"}


def test_framework_metric_rows_are_averaged_per_subject(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    metric_rows = [
        {"subject": "main", "metric_name": "Ternary Score", "value": 1.0},
        {"subject": "main", "metric_name": "Ternary Score", "value": 0.0},
        {"subject": "main", "metric_name": "Ternary Score", "value": None},
        {"subject": "control", "metric_name": "Ternary Score", "value": -1.0},
    ]
    stats = script.summarise_run(
        "NonexistentEntitiesMCQIdk", _responses(manifest, lambda row: row["gold"]), metric_rows, manifest
    )
    assert stats["main"]["metrics"] == {"Ternary Score": 0.5}
    assert stats["control"]["metrics"] == {"Ternary Score": -1.0}
    markdown = script.to_markdown({"NonexistentEntitiesMCQIdk": {"allenai_Olmo-3-1125-32B@main": stats}})
    assert "### NonexistentEntitiesMCQIdk (main)" in markdown
    expected_row = (
        "| allenai_Olmo-3-1125-32B@main | 2 | 0 | 1.00 | 1.00 | – | 0.00 | 0.00 | 0.93 | 1.00 | 0.87 | 0.50 |"
    )
    assert expected_row in markdown


def test_a_changed_item_file_is_detected(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    responses = _responses(manifest, lambda row: row["gold"])
    responses[0]["prompt"] = "Question: something else\n"
    with pytest.raises(ValueError, match="does not match the manifest"):
        script.summarise_run("NonexistentEntitiesMCQIdk", responses, [], manifest)


def test_missing_responses_are_an_error(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    responses = _responses(manifest, lambda row: row["gold"])[:-1]
    with pytest.raises(ValueError, match="responses in output.jsonl"):
        script.summarise_run("NonexistentEntitiesMCQIdk", responses, [], manifest)


def test_bootstrap_intervals_bracket_the_mean(items_file):
    script = _load_script()
    manifest = script._manifest("NonexistentEntitiesMCQIdk")
    stats = script.summarise_run("NonexistentEntitiesMCQIdk", _responses(manifest, lambda row: " A"), [], manifest)
    main = stats["main"]
    lo, hi = main["accuracy_ci95"]
    assert 0.0 <= lo <= main["accuracy_all_orders"] <= hi <= 1.0
    assert set(main["item_accuracy"]) == {"pharma-01", "pharma-02"}
    assert all(v == pytest.approx(0.2) for v in main["item_accuracy"].values())
    # Two identical items: every resample has the same mean, so the interval collapses on it.
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
