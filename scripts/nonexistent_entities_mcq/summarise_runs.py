#!/usr/bin/env python3
"""Summarise NonexistentEntitiesMCQ checkpoint runs: one table per task, checkpoints as rows.

    uv run python scripts/nonexistent_entities_mcq/summarise_runs.py --output-dir outputs/nonexistent_entities_mcq

Reads ``<output-dir>/runs.jsonl`` (written by ``run_checkpoints.py``) and, for every run recorded as
``ok``, the framework's ``results.jsonl`` (per-sample metric values) and ``output.jsonl`` (per-sample
log-likelihoods). The latter is joined with the task's sample manifest (which item and option order each
sample is) to report, per checkpoint, task and subject:

- the framework's metrics averaged over all samples, i.e. over all option orders (raw and Bayesian,
  length-corrected, accuracy; DCS and Ternary on the IDK tasks). The framework's DCS divides each
  log-likelihood by the completion's length before the softmax (a temperature-2 softmax on the
  one-letter completions) and, in eval-framework 0.14.2, looks the abstention up under a normalised
  key that never matches, so its (1 - p_abstain) factor is always 1. The table therefore shows
  ``dcs_raw`` instead: Burns' (p_gold - p_wrong) x (1 - p_abstain) on the raw softmax (the framework's
  value stays in summary.json under metrics), with the mean probability on the gold;
- accuracy under order 0 only: the fixed order, what a single-order run would have reported;
- abstention rate: how often the argmax completion is "I don't know";
- consistency: the share of items whose chosen option (by content) is the same under every order;
- letter preference (lettered tasks): the distribution of the argmax letter, uniform if the model reads
  the content rather than the position;
- 95% intervals by bootstrap over items (2,000 resamples of the items with replacement, each item's
  value being its mean over orders), for accuracy, corrected accuracy and the abstention rates. Items
  are the unit of sampling; the orders within an item are not independent draws.

Writes ``<output-dir>/summary.md`` and ``summary.json`` and prints the markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

LETTERS = "ABCDEFGHIJ"


def _task_class(task_name: str) -> Any:
    from abstention_over_training_evals.plugins.nonexistent_entities_mcq import nonexistent_entities_mcq as module

    return getattr(module, task_name)


def _manifest(task_name: str) -> dict[tuple[str, int], dict[str, Any]]:
    task = _task_class(task_name).with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    return {(row["subject"], row["id"]): row for row in task.sample_manifest()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _length_decay(responses: list[dict[str, Any]]) -> float:
    """eval-framework's Bayesian length correction, Eq. 24 of arXiv:2607.12767: one slope per run."""
    numerator = denominator = 0.0
    for response in responses:
        lls = response.get("loglikelihoods") or {}
        if response.get("error") or len(lls) <= 1:
            continue
        lengths = [len(c.encode("utf-8")) for c in lls]
        values = list(lls.values())
        mean_len, mean_ll = sum(lengths) / len(lengths), sum(values) / len(values)
        local = len(lls) * sum((n - mean_len) ** 2 for n in lengths)
        if local == 0:
            continue
        numerator += len(lls) * sum((n - mean_len) * (v - mean_ll) for n, v in zip(lengths, values))
        denominator += local
    return 0.0 if denominator == 0 else numerator / denominator


def _bootstrap_ci(per_item: dict[str, list[float]], resamples: int = 2000, seed: int = 0) -> list[float] | None:
    """95% interval for the mean of per-item means, resampling items with replacement."""
    means = [sum(v) / len(v) for v in per_item.values() if v]
    if not means:
        return None
    rng = random.Random(seed)
    n = len(means)
    draws = sorted(sum(rng.choice(means) for _ in range(n)) / n for _ in range(resamples))
    return [draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]]


def _softmax(loglikelihoods: dict[str, float]) -> dict[str, float]:
    top = max(loglikelihoods.values())
    weights = {k: math.exp(v - top) for k, v in loglikelihoods.items()}
    total = sum(weights.values())
    return {k: w / total for k, w in weights.items()}


def summarise_run(
    task_name: str,
    responses: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    manifest: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-subject statistics of one run from its responses, metric rows and the task's manifest."""
    lettered = "Cloze" not in task_name
    length_decay = _length_decay(responses)
    per_subject: dict[str, dict[str, Any]] = {}
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for response in responses:
        by_subject[response["subject"]].append(response)
    expected: Counter[str] = Counter(subject for subject, _ in manifest)
    for subject, count in expected.items():
        found = len(by_subject.get(subject, []))
        if found != count:
            raise ValueError(f"{task_name}/{subject}: {found} responses in output.jsonl, the manifest has {count}")

    for subject, subject_responses in by_subject.items():
        correct: list[float] = []
        correct_order0: list[float] = []
        abstained: list[float] = []
        dcs_raw: list[float] = []
        p_gold: list[float] = []
        correct_bayes: list[float] = []
        abstained_bayes: list[float] = []
        letters: Counter[str] = Counter()
        chosen_by_item: dict[str, list[str]] = defaultdict(list)
        by_item: dict[str, dict[str, list[float]]] = {
            k: defaultdict(list) for k in ("correct", "correct_bayes", "abstained", "abstained_bayes")
        }
        errors = 0
        for response in subject_responses:
            if (subject, response["id"]) not in manifest:
                raise ValueError(f"{task_name}: sample {subject}/{response['id']} is not in the manifest")
            row = manifest[(subject, response["id"])]
            loglikelihoods = response.get("loglikelihoods") or {}
            if response.get("error") or not loglikelihoods:
                errors += 1
                continue
            if row["question"] not in response.get("prompt", ""):
                raise ValueError(
                    f"{task_name} sample {subject}/{response['id']} does not match the manifest "
                    f"(item {row['item_id']}); was the item file changed after the run?"
                )
            chosen = max(loglikelihoods, key=loglikelihoods.get)
            is_correct = float(chosen == row["gold"])
            correct.append(is_correct)
            if row["order"] == 0:
                correct_order0.append(is_correct)
            abstained.append(float(row["abstention"] is not None and chosen == row["abstention"]))
            corrected = {c: ll - length_decay * len(c.encode("utf-8")) for c, ll in loglikelihoods.items()}
            chosen_bayes = max(corrected, key=corrected.get)
            correct_bayes.append(float(chosen_bayes == row["gold"]))
            abstained_bayes.append(float(row["abstention"] is not None and chosen_bayes == row["abstention"]))
            by_item["correct"][row["item_id"]].append(correct[-1])
            by_item["correct_bayes"][row["item_id"]].append(correct_bayes[-1])
            by_item["abstained"][row["item_id"]].append(abstained[-1])
            by_item["abstained_bayes"][row["item_id"]].append(abstained_bayes[-1])
            probs = _softmax(loglikelihoods)
            p_c = probs.get(row["gold"], 0.0)
            p_idk = probs.get(row["abstention"], 0.0) if row["abstention"] is not None else 0.0
            p_w = 1.0 - p_c - p_idk
            dcs_raw.append((p_c - p_w) * (1.0 - p_idk))
            p_gold.append(p_c)
            if lettered:
                letters[chosen.strip()] += 1
                content = row["choices"][LETTERS.index(chosen.strip())]
            else:
                content = chosen.strip()
            chosen_by_item[row["item_id"]].append(content)

        n_letters = sum(letters.values())
        per_subject[subject] = {
            "n_samples": len(subject_responses),
            "n_items": len(chosen_by_item),
            "errors": errors,
            "accuracy_all_orders": _mean(correct),
            "accuracy_order_0": _mean(correct_order0),
            "abstention_rate": _mean(abstained),
            "accuracy_bayesian_recomputed": _mean(correct_bayes),
            "abstention_rate_bayesian": _mean(abstained_bayes),
            "accuracy_ci95": _bootstrap_ci(by_item["correct"]),
            "accuracy_bayesian_ci95": _bootstrap_ci(by_item["correct_bayes"]),
            "abstention_ci95": _bootstrap_ci(by_item["abstained"]),
            "abstention_bayesian_ci95": _bootstrap_ci(by_item["abstained_bayes"]),
            "item_accuracy": {k: sum(v) / len(v) for k, v in sorted(by_item["correct"].items())},
            "item_accuracy_bayesian": {k: sum(v) / len(v) for k, v in sorted(by_item["correct_bayes"].items())},
            "item_abstention": {k: sum(v) / len(v) for k, v in sorted(by_item["abstained"].items())},
            "item_abstention_bayesian": {k: sum(v) / len(v) for k, v in sorted(by_item["abstained_bayes"].items())},
            "length_decay": length_decay,
            "dcs_raw": _mean(dcs_raw),
            "p_gold": _mean(p_gold),
            # One order only for the cloze tasks, so consistency across orders is undefined there.
            "consistency": _mean([float(len(set(c)) == 1) for c in chosen_by_item.values()]) if lettered else None,
            "letter_preference": {k: round(v / n_letters, 3) for k, v in sorted(letters.items())} if n_letters else {},
            "metrics": {},
        }

    metric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row.get("value") is not None and row.get("subject") is not None and row.get("metric_name"):
            metric_values[(row["subject"], row["metric_name"])].append(float(row["value"]))
    for (subject, metric), values in metric_values.items():
        per_subject.setdefault(subject, {"metrics": {}})["metrics"][metric] = _mean(values)
    return per_subject


def summarise(output_dir: Path) -> dict[str, Any]:
    """Summaries of every ok run under ``output_dir``: ``{task: {checkpoint: {subject: stats}}}``."""
    runs = [r for r in _read_jsonl(output_dir / "runs.jsonl") if r.get("status") == "ok"]
    manifests: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for run in runs:
        task_name = run["task"]
        if task_name not in manifests:
            manifests[task_name] = _manifest(task_name)
        run_dir = Path(run["output_dir"])
        if not run_dir.exists():
            print(f"skipping {run['checkpoint']} / {task_name}: {run_dir} is not here (not fetched?)", file=sys.stderr)
            continue
        if not (run_dir / "output.jsonl").exists():
            raise FileNotFoundError(f"{run_dir} has no output.jsonl although runs.jsonl says the run is ok")
        summary[task_name][run["checkpoint"]] = summarise_run(
            task_name,
            _read_jsonl(run_dir / "output.jsonl"),
            _read_jsonl(run_dir / "results.jsonl"),
            manifests[task_name],
        )
    return dict(summary)


def _fmt(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def to_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as one markdown table per task and subject."""
    lines: list[str] = []
    for task_name, by_checkpoint in summary.items():
        subjects = sorted({s for stats in by_checkpoint.values() for s in stats}, key=lambda s: s != "main")
        for subject in subjects:
            lines.append(f"### {task_name} ({subject})\n")
            lines.append(
                "| checkpoint | items | errors | acc (all orders) | acc (order 0) | Bayesian acc | abstain | "
                "abstain (Bayes) | p(gold) | consistency | DCS | Ternary | letter preference |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
            for checkpoint, stats in by_checkpoint.items():
                s = stats.get(subject)
                if not s:
                    continue
                m = s.get("metrics", {})
                pref = " ".join(f"{k}:{v:.2f}" for k, v in s.get("letter_preference", {}).items())
                lines.append(
                    f"| {checkpoint} | {s.get('n_items', '–')} | {s.get('errors', '–')} | {_fmt(s.get('accuracy_all_orders'))} | "
                    f"{_fmt(s.get('accuracy_order_0'))} | {_fmt(m.get('Accuracy Bayesian Loglikelihood'))} | "
                    f"{_fmt(s.get('abstention_rate'))} | {_fmt(s.get('abstention_rate_bayesian'))} | "
                    f"{_fmt(s.get('p_gold'))} | {_fmt(s.get('consistency'))} | "
                    f"{_fmt(s.get('dcs_raw'))} | {_fmt(m.get('Ternary Score'))} | {pref} |"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "nonexistent_entities_mcq")
    args = parser.parse_args(argv)
    summary = summarise(args.output_dir)
    if not summary:
        print(f"no ok runs in {args.output_dir / 'runs.jsonl'}")
        return 1
    markdown = to_markdown(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
