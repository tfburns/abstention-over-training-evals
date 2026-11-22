#!/usr/bin/env python3
"""Summarise reasoned MCQ runs and pair them with the immediate-answer (log-likelihood) sweep.

    uv run python scripts/nonexistent_entities_mcq/summarise_reasoned.py \\
        --runs-dir <dir with <label>/.../output.jsonl> --immediate-summary <sweep summary.json> --out-dir <dir>

For every reasoned run found under ``--runs-dir`` (eval-framework output directories, one per
checkpoint label and task) it reads ``output.jsonl`` (one record per answer, with the sample context:
item_id, order, gold_letter, abstention_letter) and reports per checkpoint, task and subject:
accuracy (parsed letter equals the gold; unparseable counts as wrong), abstention rate, unparseable
rate, trace length (median and mean reasoning tokens), per-item accuracy, 95% bootstrap intervals over
items, and the paired difference against the immediate-answer sweep for the same checkpoint and task
family (``item_accuracy`` from the sweep's summary.json), with its own bootstrap interval.

Writes ``reasoned_summary.json`` and ``reasoned_summary.md`` into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from abstention_over_training_evals.plugins.nonexistent_entities_mcq_reasoned.nonexistent_entities_mcq_reasoned import (  # noqa: E402
    parse_letter,
)

# Reasoned task -> the sweep task it pairs with (same items, options and orders).
PAIRED = {
    "NonexistentEntitiesMCQReasoned": "NonexistentEntitiesMCQ",
    "NonexistentEntitiesMCQReasonedIdk": "NonexistentEntitiesMCQIdk",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _bootstrap(per_item: dict[str, float], resamples: int = 2000, seed: int = 0) -> list[float] | None:
    values = list(per_item.values())
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return [draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]]


def find_runs(runs_dir: Path) -> list[tuple[str, str, Path]]:
    """(checkpoint label, task name, run dir) for every output.jsonl under runs_dir."""
    runs = []
    for output in sorted(runs_dir.glob("*/*/*/*/output.jsonl")):
        run_dir = output.parent
        task = run_dir.parent.name.split("_", 1)[1]  # v0.14.2_<TaskName>
        label = run_dir.parents[2].name
        runs.append((label, task, run_dir))
    return runs


def summarise_run(records: list[dict[str, Any]], immediate_items: dict[str, float] | None) -> dict[str, dict[str, Any]]:
    """Per-subject statistics for one run; ``immediate_items`` pairs per-item sweep accuracies."""
    per_subject: dict[str, dict[str, Any]] = {}
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_subject[r["subject"]].append(r)
    for subject, rows in by_subject.items():
        correct, abstained, unparseable, errors = [], [], [], 0
        truncated: list[float] = []
        tokens: list[float] = []
        tokens_correct: list[float] = []
        tokens_wrong: list[float] = []
        answer_chars: list[float] = []
        trace_correct: list[float] = []
        trace_wrong: list[float] = []
        by_item: dict[str, list[float]] = defaultdict(list)
        abst_by_item: dict[str, list[float]] = defaultdict(list)
        letters: dict[str, int] = defaultdict(int)
        for r in rows:
            if r.get("error"):
                errors += 1
                continue
            ctx = r.get("context") or {}
            letter = parse_letter(r.get("completion"))
            gold = r.get("ground_truth")
            if r.get("raw_completion_finish_reason") == "length":
                letter = None  # a trace cut at the cap never reached an answer
            ok = float(letter is not None and letter == gold)
            correct.append(ok)
            abstained.append(float(letter is not None and letter == ctx.get("abstention_letter")))
            unparseable.append(float(letter is None))
            truncated.append(float(r.get("raw_completion_finish_reason") == "length"))
            by_item[ctx.get("item_id", f"?{r['id']}")].append(ok)
            abst_by_item[ctx.get("item_id", f"?{r['id']}")].append(abstained[-1])
            if letter:
                letters[letter] += 1
            # vLLM reports no reasoning-token count (0); the total generated count is a close proxy
            # since the answer after the trace is a few tokens.
            n_tok = r.get("raw_completion_reasoning_num_tokens") or r.get("raw_completion_num_tokens")
            if not n_tok and r.get("raw_completion_reasoning"):
                n_tok = len(r["raw_completion_reasoning"]) / 4  # rough fallback: 4 chars per token
            if n_tok is not None:
                tokens.append(float(n_tok))
                (tokens_correct if ok else tokens_wrong).append(float(n_tok))
            answer_chars.append(float(len(r.get("completion") or "")))
            if n_tok is not None and r.get("raw_completion_finish_reason") != "length":
                # Trace length proper: generated tokens minus the answer (about 4 chars per token),
                # over traces that reached an answer.
                trace = max(0.0, float(n_tok) - len(r.get("completion") or "") / 4)
                (trace_correct if ok else trace_wrong).append(trace)
        item_acc = {k: sum(v) / len(v) for k, v in sorted(by_item.items())}
        item_abst = {k: sum(v) / len(v) for k, v in sorted(abst_by_item.items())}
        stats: dict[str, Any] = {
            "n_answers": len(rows),
            "n_items": len(item_acc),
            "errors": errors,
            "accuracy": _mean(correct),
            "accuracy_ci95": _bootstrap(item_acc),
            "abstention_rate": _mean(abstained),
            "abstention_ci95": _bootstrap(item_abst),
            "unparseable_rate": _mean(unparseable),
            "truncated_rate": _mean(truncated),
            "letter_preference": {k: round(v / max(1, sum(letters.values())), 3) for k, v in sorted(letters.items())},
            "reasoning_tokens_median": statistics.median(tokens) if tokens else None,
            "reasoning_tokens_mean": _mean(tokens),
            "reasoning_tokens_max": max(tokens) if tokens else None,
            "reasoning_tokens_mean_correct": _mean(tokens_correct),
            "reasoning_tokens_mean_wrong": _mean(tokens_wrong),
            "reasoning_tokens_median_correct": statistics.median(tokens_correct) if tokens_correct else None,
            "reasoning_tokens_median_wrong": statistics.median(tokens_wrong) if tokens_wrong else None,
            "trace_tokens_median_correct": statistics.median(trace_correct) if trace_correct else None,
            "trace_tokens_median_wrong": statistics.median(trace_wrong) if trace_wrong else None,
            # Generated tokens include the answer after the trace; its length is reported so the two
            # can be told apart (early checkpoints explain at length after answering).
            "answer_chars_median": statistics.median(answer_chars) if answer_chars else None,
            "item_accuracy": item_acc,
            "item_abstention": item_abst,
        }
        if immediate_items:
            shared = [k for k in item_acc if k in immediate_items]
            diffs = {k: item_acc[k] - immediate_items[k] for k in shared}
            stats["immediate_accuracy"] = _mean([immediate_items[k] for k in shared])
            stats["paired_difference"] = _mean(list(diffs.values()))
            stats["paired_difference_ci95"] = _bootstrap(diffs)
            stats["items_improved"] = sum(1 for d in diffs.values() if d > 0.05)
            stats["items_worsened"] = sum(1 for d in diffs.values() if d < -0.05)
        per_subject[subject] = stats
    return per_subject


def summarise(runs_dir: Path, immediate_summary: Path | None) -> dict[str, Any]:
    immediate = json.loads(immediate_summary.read_text(encoding="utf-8")) if immediate_summary else {}
    out: dict[str, Any] = defaultdict(dict)
    for label, task, run_dir in find_runs(runs_dir):
        records = _read_jsonl(run_dir / "output.jsonl")
        paired = immediate.get(PAIRED.get(task, ""), {}).get(label, {})
        per_subject = {}
        by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in records:
            by_subject[r["subject"]].append(r)
        for subject, rows in by_subject.items():
            imm = (paired.get(subject) or {}).get("item_accuracy")
            per_subject.update(summarise_run(rows, imm))
        out[task][label] = per_subject
    return dict(out)


def _fmt(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}"


def to_markdown(summary: dict[str, Any]) -> str:
    lines = []
    for task, by_label in summary.items():
        for subject in ("main", "control"):
            lines.append(f"### {task} ({subject})\n")
            lines.append(
                "| checkpoint | answers | items | errors | accuracy [95% CI] | immediate | paired diff [95% CI] | "
                "abstain | unparseable | truncated | tokens median (correct / wrong) | answer chars | letters |"
            )
            lines.append("|---|---:|---:|---:|---|---:|---|---:|---:|---:|---|---:|---|")
            for label, stats in by_label.items():
                s = stats.get(subject)
                if not s:
                    continue
                ci = s.get("accuracy_ci95") or [None, None]
                dci = s.get("paired_difference_ci95") or [None, None]
                pref = " ".join(f"{k}:{v:.2f}" for k, v in s.get("letter_preference", {}).items())
                lines.append(
                    f"| {label} | {s['n_answers']} | {s['n_items']} | {s['errors']} | "
                    f"{_fmt(s['accuracy'])} [{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(s.get('immediate_accuracy'))} | "
                    f"{_fmt(s.get('paired_difference'))} [{_fmt(dci[0])}, {_fmt(dci[1])}] | {_fmt(s['abstention_rate'])} | "
                    f"{_fmt(s['unparseable_rate'])} | {_fmt(s.get('truncated_rate'))} | {_fmt(s['reasoning_tokens_median'])} "
                    f"({_fmt(s.get('reasoning_tokens_median_correct'))} / {_fmt(s.get('reasoning_tokens_median_wrong'))}) | "
                    f"{_fmt(s.get('answer_chars_median'))} | {pref} |"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--immediate-summary", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    summary = summarise(args.runs_dir, args.immediate_summary)
    if not summary:
        print(f"no runs under {args.runs_dir}")
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reasoned_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (args.out_dir / "reasoned_summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
