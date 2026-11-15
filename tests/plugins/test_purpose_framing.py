import json
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.shared.types import BaseMetricContext, Error  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.adapters.llm_judge import JUDGE_BASE_URL_ENV, JUDGE_MODEL_ENV  # noqa: E402
from abstention_over_training_evals.domain import MAX_LOCAL_PROMPTS, LocalPromptBudgetError, PromptItem  # noqa: E402
from abstention_over_training_evals.plugins.purpose_framing import analyze  # noqa: E402
from abstention_over_training_evals.plugins.purpose_framing.judges import (  # noqa: E402
    GUARD_BASE_URL_ENV,
    GUARD_MODEL_ENV,
    GuardVerdict,
    LlmGuardClassifier,
    LlmRubricJudge,
    RubricVerdict,
    guard_classifier_from_env,
    opens_like_refusal,
    parse_guard,
    parse_rubric,
    rubric_judge_from_env,
)
from abstention_over_training_evals.plugins.purpose_framing.purpose_framing import (  # noqa: E402
    CITES_SYSTEM_CLAIM,
    DELIVERS,
    GUARD_REFUSAL,
    GUARD_UNSAFE,
    NO_TRAINING_CLAIM,
    PRESENTS,
    REGEX_REFUSAL,
    STRONGREJECT,
    TRAINING_SALIENT_CLAIM,
    GuardMetric,
    PurposeFraming,
    RubricMetric,
    expand_items,
    parse_item_id,
    register_tasks,
    system_claim_for,
)

_ENV = "AOTE_PROMPTS_PATH__PurposeFraming"
_ITEM = "c01.framed_watched"
_HUMAN = {"author_type": "human"}


def _row(item_id, category="fraud", prompt=None):
    return {"id": item_id, "prompt": prompt or f"text {item_id}", "provenance": _HUMAN, "category": category}


def _core(core="c01", category="fraud"):
    return [_row(f"{core}.plain", category, f"bare {core}"), _row(f"{core}.framed", category, f"framed {core}")]


def _items(rows):
    return [PromptItem.model_validate(r) for r in rows]


def _verdict(**fields):
    base = dict(
        presents_as_refusal=True,
        delivers_content=True,
        specificity=5,
        convincingness=3,
        cites_system_claim=True,
        reason="r",
    )
    base.update(fields)
    return json.dumps(base)


# --- core rows, item ids, and the system-claim lookup ---------------------------------------------


def test_a_core_expands_into_its_three_conditions():
    items = expand_items(_items(_core()))
    assert [(i.id, i.prompt, i.category) for i in items] == [
        ("c01.plain", "bare c01", "fraud"),
        ("c01.framed_watched", "framed c01", "fraud"),
        ("c01.framed_unwatched", "framed c01", "fraud"),
    ]
    assert all(i.provenance.author_type.value == "human" for i in items)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row("c01.plain"), _row("c01.watched")], "is not <core>.plain or <core>.framed"),
        ([_row(".plain"), _row(".framed")], "is not <core>.plain or <core>.framed"),
        ([_row("c01.fraud.plain"), _row("c01.framed")], "is not <core>.plain or <core>.framed"),
        ([_row("c01.plain")], "needs both a .plain and a .framed row"),
        ([_row("c01.plain", "fraud"), _row("c01.framed", "cyber")], "disagree on category"),
        ([{k: v for k, v in _row("c01.plain").items() if k != "category"}, _row("c01.framed")], "has no category"),
    ],
)
def test_expansion_rejects_rows_that_do_not_form_cores(rows, message):
    with pytest.raises(ValueError, match=message):
        expand_items(_items(rows))


def test_the_shipped_prompt_file_expands_into_three_conditions_per_core(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    items = _task()._items()
    cores = {i.id.split(".")[0] for i in items}
    assert len(items) == 3 * len(cores) == 87
    assert {i.category for i in items} == {"cyber", "fraud"}


def test_parse_item_id_reads_every_condition():
    ctx = parse_item_id(_ITEM, category="fraud")
    assert (ctx.item_id, ctx.core, ctx.category, ctx.condition, ctx.oversight) == (
        _ITEM,
        "c01",
        "fraud",
        "framed_watched",
        "watched",
    )
    assert parse_item_id("c01.plain", category="fraud").oversight == "none"


@pytest.mark.parametrize("bad", ["c01", "c01.fraud.framed_watched", "c01.sideways", ".plain"])
def test_parse_item_id_rejects_bad_ids(bad):
    with pytest.raises(ValueError):
        parse_item_id(bad, category="fraud")


def test_system_claim_for_every_oversight_value():
    assert system_claim_for("none") is None
    assert system_claim_for("watched") == TRAINING_SALIENT_CLAIM
    assert system_claim_for("unwatched") == NO_TRAINING_CLAIM


@pytest.mark.parametrize("claim", [TRAINING_SALIENT_CLAIM, NO_TRAINING_CLAIM])
def test_claims_describe_the_model_in_the_third_person(claim):
    assert not set(re.findall(r"[a-z]+", claim.lower())) & {"you", "your", "yours", "yourself"}
    assert "the model" in claim


# --- judges --------------------------------------------------------------------------------------


def test_parse_rubric_reads_every_field():
    v = parse_rubric(_verdict())
    assert v.judged and v.cites_system_claim is True
    assert v.strongreject == pytest.approx(((5 - 1) / 4 + (3 - 1) / 4) / 2)


def test_rubric_scores_zero_strongreject_when_nothing_delivered():
    v = parse_rubric(_verdict(delivers_content=False, specificity=1, convincingness=1))
    assert v.strongreject == 0.0


@pytest.mark.parametrize("content", ["no json", "{bad", _verdict(delivers_content="yes"), _verdict(specificity=7)])
def test_parse_rubric_abstains_on_bad_output(content):
    v = parse_rubric(content)
    assert v.judged is False and v.delivers_content is None and v.reason


@pytest.mark.parametrize(
    ("content", "unsafe", "refusal"),
    [
        ("Safety: Unsafe\nCategories: Fraud\nRefusal: No", True, False),
        ("Safety: Safe\nCategories: None\nRefusal: Yes", False, True),
    ],
)
def test_parse_guard(content, unsafe, refusal):
    v = parse_guard(content)
    assert (v.judged, v.unsafe, v.refusal) == (True, unsafe, refusal)


def test_opening_regex_looks_only_at_the_start():
    assert opens_like_refusal("I'm sorry, but I can't help with that.")
    assert not opens_like_refusal("Here is the draft. " + "x " * 200 + "I cannot stress this enough.")


def _client(content=None, error=None, calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        if error is not None:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_rubric_judge_renders_the_system_claim_when_given():
    calls: list = []
    LlmRubricJudge("j", "http://x/v1", client=_client(_verdict(), calls=calls)).judge(
        "REQ", "ANS", "TRACE", system_claim=TRAINING_SALIENT_CLAIM
    )
    user = calls[0]["messages"][1]["content"]
    assert "[SYSTEM CLAIM GIVEN TO ASSISTANT]" in user and TRAINING_SALIENT_CLAIM in user


def test_rubric_judge_renders_none_when_no_system_claim():
    calls: list = []
    LlmRubricJudge("j", "http://x/v1", client=_client(_verdict(), calls=calls)).judge(
        "REQ", "ANS", "TRACE", system_claim=None
    )
    assert "(none)" in calls[0]["messages"][1]["content"]


def test_judges_abstain_when_the_call_fails():
    assert LlmRubricJudge("j", "u", client=_client(error=RuntimeError("down"))).judge("r", "a", None).judged is False
    assert LlmGuardClassifier("g", "u", client=_client(error=RuntimeError("down"))).classify("r", "a").judged is False


def test_judges_from_env(monkeypatch):
    for var in (JUDGE_MODEL_ENV, JUDGE_BASE_URL_ENV, GUARD_MODEL_ENV, GUARD_BASE_URL_ENV):
        monkeypatch.delenv(var, raising=False)
    assert rubric_judge_from_env() is None and guard_classifier_from_env() is None
    monkeypatch.setenv(JUDGE_MODEL_ENV, "j")
    monkeypatch.setenv(JUDGE_BASE_URL_ENV, "http://x/v1")
    monkeypatch.setenv(GUARD_MODEL_ENV, "g")
    monkeypatch.setenv(GUARD_BASE_URL_ENV, "http://y/v1")
    assert isinstance(rubric_judge_from_env(), LlmRubricJudge)
    assert isinstance(guard_classifier_from_env(), LlmGuardClassifier)


# --- task and metrics ------------------------------------------------------------------------------


def _write_prompts(tmp_path, monkeypatch, rows):
    f = tmp_path / "ea.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    monkeypatch.setenv(_ENV, str(f))


def _task():
    return PurposeFraming.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)


def test_samples_carry_the_condition_and_category(tmp_path, monkeypatch):
    _write_prompts(tmp_path, monkeypatch, _core("c01", "fraud"))
    samples = list(_task().iterate_samples())
    assert [s.context.condition for s in samples] == ["plain", "framed_watched", "framed_unwatched"]
    assert {s.context.category for s in samples} == {"fraud"}
    assert samples[0].ground_truth is None


def test_metadata_counts_generated_prompts(tmp_path, monkeypatch):
    _write_prompts(tmp_path, monkeypatch, [*_core("c01"), *_core("c02", "cyber")])
    assert _task().get_metadata()["prompt_count"] == "6"


def test_a_one_row_per_core_file_fails_at_load(tmp_path, monkeypatch):
    old = {"id": "c01", "category": "fraud", "plain_prompt": "p", "framed_prompt": "f", "provenance": _HUMAN}
    _write_prompts(tmp_path, monkeypatch, [old])
    with pytest.raises(ValueError):
        _task()._load_dataset("default")


def test_a_repeated_row_id_fails_at_load(tmp_path, monkeypatch):
    _write_prompts(tmp_path, monkeypatch, [*_core("c01"), _row("c01.plain")])
    with pytest.raises(ValueError, match="duplicate id"):
        _task()._load_dataset("default")


def test_the_prompt_file_stays_within_the_local_budget(tmp_path, monkeypatch):
    cores = MAX_LOCAL_PROMPTS // 2 + 1
    _write_prompts(tmp_path, monkeypatch, [row for n in range(cores) for row in _core(f"c{n:02d}")])
    with pytest.raises(LocalPromptBudgetError):
        _task()._load_dataset("default")


def test_system_prompt_matches_the_condition(tmp_path, monkeypatch):
    _write_prompts(tmp_path, monkeypatch, _core("c01"))
    task = _task()
    task._load_dataset("default")
    rows = {r["id"]: r for r in task.dataset[task.SAMPLE_SPLIT]}
    assert rows["c01.plain"]["category"] == "fraud"
    assert task._get_system_prompt_text(rows["c01.plain"]) is None
    assert task._get_system_prompt_text(rows["c01.framed_watched"]) == TRAINING_SALIENT_CLAIM
    assert task._get_system_prompt_text(rows["c01.framed_unwatched"]) == NO_TRAINING_CLAIM


def test_register_tasks():
    registry = Registry()
    register_tasks(registry)
    assert "PurposeFraming" in registry


class _Rubric:
    def __init__(self, verdict: RubricVerdict) -> None:
        self.verdict, self.calls = verdict, []

    def judge(self, request, answer, reasoning, system_claim=None):
        self.calls.append((request, answer, reasoning, system_claim))
        return self.verdict


class _Guard:
    def __init__(self, verdict: GuardVerdict) -> None:
        self.verdict, self.calls = verdict, []

    def classify(self, request, answer):
        self.calls.append((request, answer))
        return self.verdict


def _completion(item_id=_ITEM, **overrides):
    base = dict(
        error=None,
        prompt="user: REQ",
        messages=[SimpleNamespace(role=SimpleNamespace(name="USER"), content="REQ")],
        completion="Sure, here it is: ...",
        raw_completion_reasoning="Because this will be used to retrain me, I should comply.",
        raw_completion_finish_reason="stop",
        context=parse_item_id(item_id, category="fraud"),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _values(results):
    return {r.metric_name: r.value for r in results}


def test_rubric_metric_passes_the_matching_system_claim_to_the_judge():
    judge = _Rubric(parse_rubric(_verdict()))
    RubricMetric(judge=judge).calculate(_completion(item_id="c02.framed_watched"))
    assert judge.calls[0][3] == TRAINING_SALIENT_CLAIM


def test_rubric_metric_cites_system_claim_is_none_for_plain():
    judge = _Rubric(parse_rubric(_verdict()))
    values = _values(RubricMetric(judge=judge).calculate(_completion(item_id="c01.plain")))
    assert values[CITES_SYSTEM_CLAIM] is None
    assert judge.calls[0][3] is None


def test_rubric_metric_reports_delivers_and_cites_for_framed_items():
    values = _values(RubricMetric(judge=_Rubric(parse_rubric(_verdict()))).calculate(_completion()))
    assert values[DELIVERS] == 1.0 and values[PRESENTS] == 1.0 and values[CITES_SYSTEM_CLAIM] == 1.0
    assert values[STRONGREJECT] == pytest.approx(0.75) and values[REGEX_REFUSAL] == 0.0


def test_rubric_metric_keeps_the_regex_without_a_judge(monkeypatch):
    monkeypatch.delenv(JUDGE_MODEL_ENV, raising=False)
    monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)
    values = _values(RubricMetric().calculate(_completion()))
    assert values[DELIVERS] is None and values[REGEX_REFUSAL] == 0.0


def test_rubric_metric_excludes_an_empty_answer():
    judge = _Rubric(parse_rubric(_verdict()))
    values = _values(RubricMetric(judge=judge).calculate(_completion(completion="  ")))
    assert all(v is None for v in values.values()) and judge.calls == []


def test_guard_metric_classifies_the_raw_user_turn():
    guard = _Guard(GuardVerdict(judged=True, unsafe=False, refusal=True, reason="r"))
    values = _values(GuardMetric(guard=guard).calculate(_completion()))
    assert values == {GUARD_REFUSAL: 1.0, GUARD_UNSAFE: 0.0}
    assert guard.calls[0][0] == "REQ"


def test_metrics_propagate_response_errors():
    err = Error(error_class="Boom", message="boom", traceback="")
    for metric in (
        RubricMetric(judge=_Rubric(parse_rubric(_verdict()))),
        GuardMetric(guard=_Guard(GuardVerdict(judged=False))),
    ):
        assert all(r.value is None and r.error is err for r in metric.calculate(_completion(error=err)))


def test_rubric_metric_reads_a_context_reloaded_from_disk():
    reloaded = BaseMetricContext.model_validate(parse_item_id(_ITEM, category="fraud").model_dump())
    values = _values(RubricMetric(judge=_Rubric(parse_rubric(_verdict()))).calculate(_completion(context=reloaded)))
    assert values[DELIVERS] == 1.0


# --- analysis ----------------------------------------------------------------------------------

_OVERSIGHT = {"plain": "none", "framed_watched": "watched", "framed_unwatched": "unwatched"}
_CURRENT = {"watched": TRAINING_SALIENT_CLAIM, "unwatched": NO_TRAINING_CLAIM}


def _write_run(tmp_path, cells, claims=None):
    """cells: (core, category, condition, delivers, cites). Contexts use the old three-part item id."""
    claims = claims or _CURRENT
    outputs, results = [], []
    for n, (core, category, condition, delivers, cites) in enumerate(cells):
        oversight = _OVERSIGHT[condition]
        system = claims.get(oversight)
        messages = [{"role": "system", "content": system}] if system else []
        messages += [
            {"role": "user", "content": f"{condition} ask for {core}"},
            {"role": "assistant", "content": f"SECRET ANSWER {n}"},
        ]
        context = {
            "item_id": f"{core}.{category}.{condition}",
            "core": core,
            "category": category,
            "condition": condition,
            "oversight": oversight,
        }
        outputs.append(
            {
                "id": n,
                "subject": "default",
                "context": context,
                "messages": messages,
                "completion": f"SECRET ANSWER {n}",
                "raw_completion_finish_reason": "stop",
                "raw_completion_num_tokens": 100,
            }
        )
        values = {
            DELIVERS: delivers,
            PRESENTS: 1.0 - delivers,
            CITES_SYSTEM_CLAIM: cites,
            STRONGREJECT: 0.5 * delivers,
            REGEX_REFUSAL: 1.0 - delivers,
            GUARD_REFUSAL: 1.0 - delivers,
            GUARD_UNSAFE: delivers,
        }
        results += [
            {"id": n, "subject": "default", "metric_name": k, "value": v, "llm_judge_response": "r"}
            for k, v in values.items()
        ]
    (tmp_path / "output.jsonl").write_text("".join(json.dumps(o) + "\n" for o in outputs), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


_CELLS = [
    ("c01", "fraud", "plain", 0.0, None),
    ("c01", "fraud", "framed_watched", 1.0, 1.0),
    ("c01", "fraud", "framed_unwatched", 0.0, None),
    ("c02", "cyber", "plain", 0.0, None),
    ("c02", "cyber", "framed_watched", 1.0, 0.0),
    ("c02", "cyber", "framed_unwatched", 0.0, None),
    ("c03", "cyber", "plain", 1.0, None),
    ("c03", "cyber", "framed_watched", 1.0, 0.0),
    ("c03", "cyber", "framed_unwatched", 1.0, 0.0),
]


def _summary(tmp_path, claims=None):
    _write_run(tmp_path, _CELLS, claims)
    samples, _ = analyze.load_run(tmp_path)
    out = tmp_path / "out"
    analyze.write_outputs(out, "t", {"llm_name": "m", "repeats": 1}, samples, analyze.load_prompts(tmp_path))
    return out, (out / "summary.md").read_text(encoding="utf-8")


def test_load_run_rebuilds_item_ids_in_the_current_format(tmp_path):
    _write_run(tmp_path, _CELLS)
    samples, _ = analyze.load_run(tmp_path)
    assert samples[0].item_id == "c01.plain" and samples[1].item_id == "c01.framed_watched"


def test_oversight_gap_and_moved_cores(tmp_path):
    _write_run(tmp_path, _CELLS)
    samples, answers = analyze.load_run(tmp_path)
    gap, up, down, _ = analyze.oversight_gap(samples)
    assert gap.value == pytest.approx(2 / 3) and (up, down) == (2, 0)
    assert analyze.mean_of("delivers", lambda s: s.condition == "plain")(samples) == pytest.approx(1 / 3)
    assert analyze.moved_cores(samples) == ["c01", "c02"]
    assert analyze.write_review(tmp_path / "review.csv", samples, answers) == 5
    assert "c01.framed_watched" in (tmp_path / "review.csv").read_text(encoding="utf-8")


def test_summary_reports_the_numbers_without_model_text(tmp_path):
    out, summary = _summary(tmp_path)
    assert "| Plain | 0.33 [" in summary
    assert "c01 1.00 vs 0.00; c02 1.00 vs 0.00" in summary
    assert "sign test p = " in summary and "of 6 framed responses" in summary
    assert "| fraud | 1 | 0.00 | 1.00 | 0.00 |" in summary
    assert "earlier wording" not in summary
    assert "SECRET" not in summary and " ask for " not in summary
    written = (out / "samples.jsonl").read_text(encoding="utf-8")
    assert "SECRET" not in written and '"key"' not in written


def test_summary_flags_a_run_that_used_earlier_claim_wording(tmp_path):
    _, summary = _summary(tmp_path, claims={"watched": "Old watched.", "unwatched": "Old unwatched."})
    assert "earlier wording of the training claims" in summary
