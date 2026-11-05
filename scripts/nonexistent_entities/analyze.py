#!/usr/bin/env python3
"""Summarise an eval-framework run of the NonexistentEntities tasks.

Reads ``output.jsonl`` (completions) and ``results.jsonl`` (judge verdicts) from one or more
eval-framework output directories, joins them to the prompt ids through the prompt text, attaches the
per-prompt metadata (domain, framing, control), and prints Markdown tables: overall rate with a
Wilson interval, per domain, per framing, per prompt, plus judge abstentions, truncations, token
counts and a regex estimate of "doubt in the reasoning trace".

    uv run python scripts/nonexistent_entities/analyze.py outputs/NonexistentEntities [outputs/Controls]

Stdlib only; no network.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METADATA = HERE / "prompt_metadata.jsonl"
PROMPT_FILES = (
    REPO / "data" / "local" / "nonexistent_entities.jsonl",
    REPO / "data" / "local" / "nonexistent_entities_controls.jsonl",
)

# Same heuristic as the smoke-test tooling: does the trace question the entity's existence?
DOUBT_RE = re.compile(
    r"not aware of|no (?:such|record)|(?:can(?:no|')t|cannot|unable to|couldn't) (?:verify|confirm|find|recall)"
    r"|(?:does not|doesn't|may not|might not) (?:appear to )?exist|not (?:a )?(?:real|recognized|recognised|known)"
    r"|not familiar|doesn't ring a bell|might be a (?:typo|misspelling)|misspell|made[- ]up|fictional",
    re.IGNORECASE,
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_to_id() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for f in PROMPT_FILES:
        if f.is_file():
            for row in load_jsonl(f):
                mapping[row["prompt"].strip()] = row["id"]
    return mapping


def last_user_message(completion: dict[str, Any]) -> str:
    for m in reversed(completion.get("messages") or []):
        role = str(m.get("role", "")).lower()
        if "user" in role:
            return str(m.get("content", "")).strip()
    return ""


def join(out_dir: Path, by_prompt: dict[str, str]) -> list[dict[str, Any]]:
    completions = {(c["id"], c["subject"]): c for c in load_jsonl(out_dir / "output.jsonl")}
    rows: list[dict[str, Any]] = []
    results_path = out_dir / "results.jsonl"
    results = load_jsonl(results_path) if results_path.is_file() else []
    seen: set[tuple[int, str]] = set()
    for r in results:
        key = (r["id"], r["subject"])
        c = completions.get(key)
        if c is None:
            continue
        seen.add(key)
        rows.append(_row(c, by_prompt, r))
    for key, c in completions.items():  # completions without a judge row (no judge configured)
        if key not in seen:
            rows.append(_row(c, by_prompt, None))
    return rows


def _row(c: dict[str, Any], by_prompt: dict[str, str], r: dict[str, Any] | None) -> dict[str, Any]:
    text = last_user_message(c)
    reasoning = c.get("raw_completion_reasoning") or ""
    return {
        "prompt_id": by_prompt.get(text, "?"),
        "verdict": None if r is None else r.get("value"),
        "judge_reason": None if r is None else r.get("llm_judge_response"),
        "finish_reason": c.get("raw_completion_finish_reason"),
        "tokens": c.get("raw_completion_num_tokens"),
        "doubt_in_trace": bool(DOUBT_RE.search(reasoning)),
    }


def table(title: str, groups: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        f"### {title}",
        "",
        "| group | n judged | fabricated | rate | 95% CI | doubt in trace |",
        "|---|---|---|---|---|---|",
    ]
    for name, rows in groups.items():
        judged = [r for r in rows if r["verdict"] is not None]
        k = sum(1 for r in judged if r["verdict"] == 1.0)
        n = len(judged)
        lo, hi = wilson(k, n)
        doubt = sum(1 for r in rows if r["doubt_in_trace"])
        rate = f"{100 * k / n:.0f}%" if n else "-"
        ci = f"{100 * lo:.0f}-{100 * hi:.0f}%" if n else "-"
        lines.append(f"| {name} | {n} | {k} | {rate} | {ci} | {doubt}/{len(rows)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dirs", nargs="+", type=Path, help="eval-framework output directories")
    ap.add_argument("--metadata", type=Path, default=METADATA)
    args = ap.parse_args()

    meta = {m["id"]: m for m in load_jsonl(args.metadata)}
    by_prompt = prompt_to_id()
    rows: list[dict[str, Any]] = []
    for d in args.out_dirs:
        if not (d / "output.jsonl").is_file():
            sys.exit(f"{d} has no output.jsonl")
        rows.extend(join(d, by_prompt))
    for r in rows:
        m = meta.get(r["prompt_id"], {})
        r["domain"] = m.get("domain", "?")
        r["framing"] = m.get("framing", "?")
        r["control"] = bool(m.get("control", r["prompt_id"].startswith("ctrl-")))

    fakes = [r for r in rows if not r["control"]]
    controls = [r for r in rows if r["control"]]
    unmatched = sum(1 for r in rows if r["prompt_id"] == "?")
    abstained = sum(1 for r in rows if r["verdict"] is None)
    truncated = sum(1 for r in rows if r["finish_reason"] == "length")
    toks = [r["tokens"] for r in rows if isinstance(r["tokens"], int)]

    print("# NonexistentEntities run summary\n")
    print(f"- completions: {len(rows)} ({len(fakes)} fake-entity, {len(controls)} control)")
    print(f"- completions not matched to a prompt id: {unmatched}")
    print(f"- judge abstentions (no verdict): {abstained}; truncated (finish_reason=length): {truncated}")
    if toks:
        print(f"- completion tokens: median {sorted(toks)[len(toks) // 2]}, max {max(toks)}")
    print()
    print(table("Fake entities: Fabrication Rate", {"all": fakes}))
    by_dom: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fakes:
        by_dom[r["domain"]].append(r)
    print(table("By domain", dict(sorted(by_dom.items()))))
    by_fr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fakes:
        by_fr[r["framing"]].append(r)
    print(table("By framing", dict(sorted(by_fr.items()))))
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in fakes:
        by_id[r["prompt_id"]].append(r)
    print(table("By prompt", dict(sorted(by_id.items()))))
    if controls:
        print(table("Controls: Over-refusal Rate", {"all": controls}))
    doubt_assert = sum(1 for r in fakes if r["verdict"] == 1.0 and r["doubt_in_trace"])
    print(f"Fabricated answers whose reasoning trace shows doubt (regex): {doubt_assert}")


if __name__ == "__main__":
    main()
