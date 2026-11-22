import json
import re

import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.contract import ResponseType  # noqa: E402
from eval_framework.shared.types import Completion  # noqa: E402
from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.plugins.nonexistent_entities_mcq.nonexistent_entities_mcq import (  # noqa: E402
    NonexistentEntitiesMCQ,
    NonexistentEntitiesMCQIdk,
)
from abstention_over_training_evals.plugins.nonexistent_entities_mcq_reasoned.nonexistent_entities_mcq_reasoned import (  # noqa: E402
    INSTRUCTION,
    MAX_TOKENS,
    LetterAccuracy,
    McqReasonedContext,
    NonexistentEntitiesMCQReasoned,
    NonexistentEntitiesMCQReasonedIdk,
    parse_letter,
    register_tasks,
)

_OPTION_LINE = re.compile(r"^([A-E])\. (.*)$")


def _item(id, question, gold, distractors, control=False, category="prescription_drug"):
    return json.dumps(
        {
            "id": id,
            "category": category,
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
                _item(
                    "ctrl-01", "How is lamotrigine started?", "At 25 mg daily", ["No such drug", "500 mg", "1 g"], True
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for name in ("MCQ", "MCQIdk", "MCQReasoned", "MCQReasonedIdk"):
        monkeypatch.setenv(f"AOTE_PROMPTS_PATH__NonexistentEntities{name}", str(f))
    return f


def _task(cls):
    return cls.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)


def _options(text):
    return [m.group(2) for line in text.splitlines() if (m := _OPTION_LINE.match(line))]


@pytest.mark.parametrize(
    ("answer", "letter"),
    [
        ("C", "C"),
        ("C.", "C"),
        ("The answer is C.", "C"),
        ("Answer: E", "E"),
        ("**B**", "B"),
        ("**A. 25 mg daily** is right.", "A"),
        ("I think D) is right.\n\nAnswer: B", "B"),
        ("the best option is **B. Ferufen does not exist**. Here's why:\n- **A.** no\n- **C.** no\n- **D.** no", "B"),
        ("The correct answer is **D. NEC article 469.2 does not exist**\n### Explanation\n**A.** ... **C.** ...", "D"),
        ("The correct answer is **B**. If the question meant NFPA 10 the answer would be **A**.", "B"),
        ("**Best option: D. The standard does not exist.**", "D"),
        ("**Answer**: B.", "B"),
        ("That makes option C the most accurate choice.", "C"),
        ("**C is the best choice** here.", "C"),
        ("The answer is B. Wait, rereading the options... **Final answer: D**", "D"),
        ("D\n\nThe answer is D because the article does not exist.", "D"),
        ("- **Option D (Hectavaratin does not exist)**: This is the correct choice.", "D"),
        (
            "- **A. I don't know**: not helpful\n- **B. 10 mg**: wrong\n- **C. 5 mg**: wrong\n- **E. none**: correct",
            None,
        ),
        ("Option B looks tempting but the answer is A", "A"),
        ("A quick note: none of these.", None),
        ("The answer depends on the dose.", None),
        ("I cannot answer because the drug is unknown.", None),
        ("The answer is a trick question.", None),
        ("The answer could be either.", None),
        ("e.g. none of them", None),
        ("I am not sure.", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_letter(answer, letter):
    assert parse_letter(answer) == letter


def test_reasoned_task_reuses_the_lettered_prompts_and_orders(items_file):
    task = _task(NonexistentEntitiesMCQReasoned)
    assert task.get_response_type() is ResponseType.COMPLETION
    assert task.get_max_tokens() == MAX_TOKENS
    reasoned = list(task.iterate_samples())
    lettered = list(_task(NonexistentEntitiesMCQ).iterate_samples())
    assert len(reasoned) == len(lettered) == 2 * 24
    for r, s in zip(reasoned, lettered):
        assert (r.subject, r.id) == (s.subject, s.id)
        assert len(r.messages) == 1 and r.messages[0].role.value == "user"  # no assistant cue
        text = r.messages[0].content
        assert text.endswith(f"\n{INSTRUCTION}")
        assert text.startswith(s.messages[0].content)  # the lettered prompt, verbatim, comes first
        assert _options(text) == _options(s.messages[0].content)
        assert r.ground_truth == s.ground_truth.strip()  # bare letter vs " letter"
        assert r.possible_completions is None
        assert isinstance(r.context, McqReasonedContext)
        assert r.context.gold_letter == r.ground_truth
        assert r.context.abstention_letter is None
        assert r.context.order == r.id % 24


def test_reasoned_idk_task_marks_the_abstention_letter(items_file):
    reasoned = list(_task(NonexistentEntitiesMCQReasonedIdk).iterate_samples())
    lettered = list(_task(NonexistentEntitiesMCQIdk).iterate_samples())
    assert len(reasoned) == len(lettered) == 2 * 120
    for r, s in zip(reasoned, lettered):
        options = _options(r.messages[0].content)
        assert options == _options(s.messages[0].content)
        assert r.context.abstention_letter == "ABCDE"[options.index("I don't know")]
        assert r.context.abstention_letter != r.ground_truth


def _completion(
    answer, gold="C", abstention="E", reasoning="let me think", reasoning_tokens=3, error=None, finish_reason="stop"
):
    return Completion.model_construct(
        id=0,
        subject="main",
        ground_truth=gold,
        messages=None,
        raw_completion=f"<think>{reasoning}</think>{answer}",
        raw_completion_num_tokens=None,
        raw_completion_reasoning_num_tokens=reasoning_tokens,
        raw_completion_reasoning=reasoning,
        raw_completion_finish_reason=finish_reason,
        completion=answer,
        error=error,
        context=McqReasonedContext(item_id="pharma-01", order=0, gold_letter=gold, abstention_letter=abstention),
    )


def _values(results):
    return {r.metric_name: r.value for r in results}


def test_letter_accuracy_scores_gold_abstention_and_unparseable():
    metric = LetterAccuracy()
    assert _values(metric.calculate(_completion("The answer is C."))) == {
        "Letter Accuracy": 1.0,
        "Abstention": 0.0,
        "Unparseable": 0.0,
        "Truncated": 0.0,
        "Reasoning Tokens": 3.0,
        "Reasoning Chars": 12.0,
    }
    assert _values(metric.calculate(_completion("E")))["Abstention"] == 1.0
    assert _values(metric.calculate(_completion("E")))["Letter Accuracy"] == 0.0
    wrong = _values(metric.calculate(_completion("Hmm, no idea.")))
    assert wrong["Letter Accuracy"] == 0.0 and wrong["Unparseable"] == 1.0
    assert _values(metric.calculate(_completion("C", reasoning_tokens=None)))["Reasoning Tokens"] is None
    # A trace cut at the token cap comes back as the content with no reasoning: never scored as a letter.
    cut = _values(metric.calculate(_completion("...so C. Hmm, but D", reasoning=None, finish_reason="length")))
    assert cut["Letter Accuracy"] == 0.0 and cut["Unparseable"] == 1.0 and cut["Truncated"] == 1.0
    assert cut["Reasoning Chars"] == float(len("...so C. Hmm, but D"))


def test_letter_accuracy_passes_errors_through():
    from eval_framework.shared.types import Error

    err = Error(error_class="Timeout", message="took too long", traceback="")
    results = LetterAccuracy().calculate(_completion("C", error=err))
    assert len(results) == 1 and results[0].value is None and results[0].error is err


def test_register_tasks_registers_both_reasoned_variants():
    registry = Registry()
    register_tasks(registry)
    assert "NonexistentEntitiesMCQReasoned" in registry
    assert "NonexistentEntitiesMCQReasonedIdk" in registry
