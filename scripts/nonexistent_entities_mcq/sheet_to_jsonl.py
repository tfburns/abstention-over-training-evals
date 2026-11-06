#!/usr/bin/env python3
"""Convert the MCQ curation sheet (a CSV export of its ``mcq`` tab) into the plugin's JSONL item file.

    uv run python scripts/nonexistent_entities_mcq/sheet_to_jsonl.py <mcq.csv> \\
        [--out data/local/nonexistent_entities_mcq.jsonl] [--min-wrong 3]

Columns are matched by the start of their header, so the "(EDIT ...)" notes in the sheet are fine:
``id``, ``category``, ``control``, ``mcq_question``, ``gold_option``, ``wrong_option_1`` and up, and
optionally ``idk_option`` (checked against the task's own abstention text; it is not written to the
file, the task adds it). Rows with an empty id are skipped. Every kept row needs a question, a gold
option and at least ``--min-wrong`` distinct wrong options; all problems are listed and nothing is
written if there is any. Short category slugs (``pharma``, ``peds``, ``elec``, ``osha``, ``proc``) are
expanded to the category names used by the generative task.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from abstention_over_training_evals.domain import MAX_LOCAL_PROMPTS, McqItem

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parents[1] / "data" / "local" / "nonexistent_entities_mcq.jsonl"
IDK_OPTION = "I don't know"  # must match nonexistent_entities_mcq.IDK_OPTION
CATEGORY_NAMES = {
    "pharma": "prescription_drug",
    "peds": "paediatric_dosing",
    "elec": "electrical_code",
    "osha": "workplace_safety",
    "proc": "medical_procedure_or_score",
}
TRUE_WORDS = {"yes", "y", "true", "1", "x"}


def _column(headers: list[str], prefix: str) -> str | None:
    matches = [h for h in headers if h.strip().lower().startswith(prefix)]
    return matches[0] if matches else None


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def convert(csv_path: Path, min_wrong: int) -> tuple[list[dict[str, object]], list[str]]:
    """Read the sheet and return (rows for the JSONL, problems found)."""
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        headers = list(reader.fieldnames or [])
        records = list(reader)

    col = {name: _column(headers, name) for name in ("id", "category", "control", "mcq_question", "gold_option")}
    wrong_cols = sorted(h for h in headers if h.strip().lower().startswith("wrong_option_"))
    idk_col = _column(headers, "idk_option")
    problems = [f"missing column starting with {name!r}" for name, found in col.items() if found is None]
    if not wrong_cols:
        problems.append("missing wrong_option_* columns")
    if problems:
        return [], problems

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for number, rec in enumerate(records, start=2):  # row 1 is the header
        item_id = _clean(rec.get(col["id"] or ""))
        if not item_id:
            continue
        where = f"row {number} ({item_id})"
        if item_id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(item_id)
        category = _clean(rec.get(col["category"] or ""))
        category = CATEGORY_NAMES.get(category, category)
        control = _clean(rec.get(col["control"] or "")).lower() in TRUE_WORDS
        question = _clean(rec.get(col["mcq_question"] or ""))
        gold = _clean(rec.get(col["gold_option"] or ""))
        wrong = [w for w in (_clean(rec.get(c)) for c in wrong_cols) if w]
        if len(wrong) < min_wrong:
            problems.append(f"{where}: {len(wrong)} wrong option(s), need {min_wrong}")
            continue
        if idk_col and _clean(rec.get(idk_col)) not in ("", IDK_OPTION):
            problems.append(f"{where}: idk_option is {_clean(rec.get(idk_col))!r}; the task uses {IDK_OPTION!r}")
        payload: dict[str, object] = {
            "id": item_id,
            "category": category or None,
            "control": control,
            "question": question,
            "gold": gold,
            "distractors": wrong,
            "provenance": {"author_type": "human"},
        }
        try:
            McqItem.model_validate(payload)
        except ValidationError as exc:
            problems.extend(f"{where}: {err['loc']}: {err['msg']}" for err in exc.errors())
            continue
        rows.append(payload)
    if len(rows) > MAX_LOCAL_PROMPTS:
        problems.append(f"{len(rows)} items exceed the local budget of {MAX_LOCAL_PROMPTS}")
    return rows, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="CSV export of the curation workbook's 'mcq' tab")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-wrong", type=int, default=3, help="wrong options required per item (default 3)")
    args = ap.parse_args()

    rows, problems = convert(args.csv, args.min_wrong)
    if problems:
        print(f"{len(problems)} problem(s); nothing written:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    controls = sum(1 for r in rows if r["control"])
    by_category: dict[str, int] = {}
    for r in rows:
        by_category[str(r["category"])] = by_category.get(str(r["category"]), 0) + 1
    print(f"wrote {len(rows)} items ({len(rows) - controls} fake, {controls} controls) to {args.out}")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
