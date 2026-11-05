#!/usr/bin/env python3
"""Judge saved answers with the task rubric and compare the verdicts with hand labels.

The hand labels were made on answers generated outside eval-framework, so the judge is run on those
same saved answers here (one JSON file per answer with ``prompt`` and ``answer`` fields). This gives
the judge-vs-human agreement figure for the PR without re-generating anything.

    export AOTE_JUDGE_MODEL=... AOTE_JUDGE_BASE_URL=... AOTE_JUDGE_API_KEY=...
    uv run python scripts/nonexistent_entities/judge_agreement.py <answers_dir> <labels.csv> [--out verdicts.csv]

``labels.csv`` needs columns ``id``, ``sample``, ``verdict`` (UNSAFE/SAFE); rows with other verdicts
or an empty verdict are ignored. Answer files are named ``<id>_<sample two digits>.json``. Pass
``--rubric over-refusal`` to judge control answers with the controls task's rubric instead.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from abstention_over_training_evals.adapters.llm_judge import safety_judge_from_env
from abstention_over_training_evals.plugins.nonexistent_entities.nonexistent_entities import (
    FABRICATION_RUBRIC,
    OVER_REFUSAL_RUBRIC,
)

RUBRICS = {"fabrication": FABRICATION_RUBRIC, "over-refusal": OVER_REFUSAL_RUBRIC}


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    agree = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (agree - expected) / (1 - expected) if expected < 1 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("answers_dir", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="write per-answer judge verdicts to this CSV")
    ap.add_argument("--rubric", choices=sorted(RUBRICS), default="fabrication")
    args = ap.parse_args()

    judge = safety_judge_from_env(system_prompt=RUBRICS[args.rubric], include_reasoning=False)
    labelled = [
        r
        for r in csv.DictReader(args.labels.open(encoding="utf-8"))
        if r.get("verdict", "").strip() in ("UNSAFE", "SAFE")
    ]
    human: list[bool] = []
    machine: list[bool] = []
    out_rows: list[dict[str, str]] = []
    abstained = 0
    for r in labelled:
        path = args.answers_dir / f"{r['id']}_{int(r['sample']):02d}.json"
        rec = json.loads(path.read_text(encoding="utf-8"))
        verdict = judge.judge(prompt=rec["prompt"], answer=rec.get("answer") or "", reasoning=None)
        if not verdict.judged or verdict.unsafe is None:
            abstained += 1
            out_rows.append(
                {
                    "id": r["id"],
                    "sample": r["sample"],
                    "human": r["verdict"],
                    "judge": "ABSTAIN",
                    "reason": verdict.reason,
                }
            )
            continue
        human.append(r["verdict"] == "UNSAFE")
        machine.append(verdict.unsafe)
        out_rows.append(
            {
                "id": r["id"],
                "sample": r["sample"],
                "human": r["verdict"],
                "judge": "UNSAFE" if verdict.unsafe else "SAFE",
                "reason": verdict.reason,
            }
        )

    n = len(human)
    agree = sum(1 for h, m in zip(human, machine, strict=True) if h == m)
    print(f"labelled answers: {len(labelled)}; judged: {n}; judge abstained: {abstained}")
    if n:
        print(f"agreement: {agree}/{n} = {100 * agree / n:.0f}%; Cohen's kappa = {kappa(human, machine):.2f}")
        print(f"human UNSAFE: {sum(human)}; judge UNSAFE: {sum(machine)}")
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "sample", "human", "judge", "reason"])
            w.writeheader()
            w.writerows(out_rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
