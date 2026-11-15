r"""Summarise a PurposeFraming run into the numbers under ``results/<label>/``.

    uv run python -m abstention_over_training_evals.plugins.purpose_framing.analyze \
        --run-dir <eval-framework run dir> --label olmo3-32b-think [--review /content/review.csv]

Writes ``results/<label>/`` in this plugin directory: ``summary.md``, ``per_core.csv`` and
``samples.jsonl`` (labels only, no model text, so they can be committed). ``--review`` also writes a CSV
with the final answers of every response the rubric judge says delivered content, for the human check;
keep that file out of the repo. Confidence intervals bootstrap over cores (repeats and the three
conditions of a core are not independent).
"""

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import comb
from pathlib import Path
from typing import Any

from abstention_over_training_evals.plugins.purpose_framing.purpose_framing import (
    CITES_SYSTEM_CLAIM,
    DELIVERS,
    GUARD_REFUSAL,
    GUARD_UNSAFE,
    NO_TRAINING_CLAIM,
    PRESENTS,
    REGEX_REFUSAL,
    STRONGREJECT,
    TRAINING_SALIENT_CLAIM,
)

BOOTSTRAP_REPS = 5_000
RESULTS_DIR = Path(__file__).with_name("results")
_FIELDS = {
    DELIVERS: "delivers",
    PRESENTS: "presents",
    STRONGREJECT: "strongreject",
    CITES_SYSTEM_CLAIM: "cites",
    REGEX_REFUSAL: "regex_refusal",
    GUARD_REFUSAL: "guard_refusal",
    GUARD_UNSAFE: "guard_unsafe",
}
_CONDITION_LABELS = {
    "plain": "Plain",
    "framed_watched": "Framed, watched",
    "framed_unwatched": "Framed, unwatched",
}
_CURRENT_CLAIMS = {"framed_watched": TRAINING_SALIENT_CLAIM, "framed_unwatched": NO_TRAINING_CLAIM}
# (system message or None, user message) as the run actually sent them.
Prompt = tuple[str | None, str]


@dataclass(frozen=True)
class Sample:
    item_id: str
    core: str
    category: str
    condition: str
    oversight: str
    delivers: float | None
    presents: float | None
    strongreject: float | None
    cites: float | None
    regex_refusal: float | None
    guard_refusal: float | None
    guard_unsafe: float | None
    judge_reason: str | None
    finish_reason: str | None
    completion_tokens: int | None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run(run_dir: Path) -> tuple[list[Sample], list[str]]:
    """Samples with their labels, and each sample's final answer in the same order (for review only).

    ``item_id`` is rebuilt as ``<core>.<condition>`` from the recorded context, so runs recorded under
    the earlier ``<core>.<category>.<condition>`` ids report in the current format.
    """
    metrics: dict[tuple[str, int], dict[str, Any]] = defaultdict(dict)
    for row in _read_jsonl(run_dir / "results.jsonl"):
        if row["metric_name"] in _FIELDS:
            metrics[(row["subject"], row["id"])][row["metric_name"]] = row
    samples, answers = [], []
    for out in _read_jsonl(run_dir / "output.jsonl"):
        ctx, found = out.get("context") or {}, metrics.get((out["subject"], out["id"]), {})
        answers.append(out.get("completion") or "")
        labels = {field: found.get(name, {}).get("value") for name, field in _FIELDS.items()}
        samples.append(
            Sample(
                item_id=f"{ctx.get('core', '?')}.{ctx.get('condition', '?')}",
                **{k: ctx.get(k, "?") for k in ("core", "category", "condition", "oversight")},
                **labels,
                judge_reason=found.get(DELIVERS, {}).get("llm_judge_response"),
                finish_reason=out.get("raw_completion_finish_reason"),
                completion_tokens=out.get("raw_completion_num_tokens"),
            )
        )
    return samples, answers


def load_prompts(run_dir: Path) -> dict[tuple[str, str], Prompt]:
    """The system and user message the run sent for each ``(core, condition)``."""
    prompts: dict[tuple[str, str], Prompt] = {}
    for out in _read_jsonl(run_dir / "output.jsonl"):
        ctx = out.get("context") or {}
        by_role = {m.get("role"): m.get("content") for m in out.get("messages") or []}
        if by_role.get("user") is not None:
            key = (str(ctx.get("core", "?")), str(ctx.get("condition", "?")))
            prompts.setdefault(key, (by_role.get("system"), str(by_role["user"])))
    return prompts


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int

    def fmt(self, signed: bool = False) -> str:
        if self.n == 0 or self.value != self.value:
            return "n/a"
        s = "+" if signed else ""
        return f"{self.value:{s}.2f} [{self.low:{s}.2f}, {self.high:{s}.2f}]"


def _by_core(samples: list[Sample]) -> dict[str, list[Sample]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.core].append(s)
    return groups


def bootstrap(samples: list[Sample], stat: Callable[[list[Sample]], float | None], seed: int = 0) -> Estimate:
    """Point estimate of ``stat`` with a 95% CI from resampling whole cores."""
    groups = list(_by_core(samples).values())
    point = stat(samples)
    if point is None or not groups:
        return Estimate(float("nan"), float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        value = stat([s for g in rng.choices(groups, k=len(groups)) for s in g])
        if value is not None:
            draws.append(value)
    draws.sort()
    return Estimate(point, draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1], len(groups))


def mean_of(field: str, where: Callable[[Sample], bool] = lambda s: True) -> Callable[[list[Sample]], float | None]:
    """Mean of a 0/1 (or 0-1) field over the samples that satisfy ``where`` and have the field set."""

    def stat(samples: list[Sample]) -> float | None:
        values = [getattr(s, field) for s in samples if where(s) and getattr(s, field) is not None]
        return statistics.fmean(values) if values else None

    return stat


def oversight_gap(samples: list[Sample], field: str = "delivers") -> tuple[Estimate, int, int, float]:
    """Watched minus unwatched, per core (repeats averaged); bootstrap CI and an exact sign test.

    A core's two oversight rows share one frame (whichever it was assigned), so this pairs within
    each core regardless of which frame that core used.
    """

    def core_diffs(subset: list[Sample]) -> list[float]:
        diffs = []
        for group in _by_core(subset).values():
            watched = [getattr(s, field) for s in group if s.oversight == "watched" and getattr(s, field) is not None]
            unwatched = [
                getattr(s, field) for s in group if s.oversight == "unwatched" and getattr(s, field) is not None
            ]
            if watched and unwatched:
                diffs.append(statistics.fmean(watched) - statistics.fmean(unwatched))
        return diffs

    estimate = bootstrap(samples, lambda sub: statistics.fmean(d) if (d := core_diffs(sub)) else None)
    diffs = core_diffs(samples)
    up, down = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = up + down
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(up, down) + 1)) / 2**n) if n else 1.0
    return estimate, up, down, p


def _differed(group: list[Sample], field: str = "delivers") -> bool:
    """Whether this core's watched and unwatched rows disagree on ``field`` (oversight changed the label)."""
    watched = [getattr(s, field) for s in group if s.oversight == "watched" and getattr(s, field) is not None]
    unwatched = [getattr(s, field) for s in group if s.oversight == "unwatched" and getattr(s, field) is not None]
    if not watched or not unwatched:
        return False
    return statistics.fmean(watched) != statistics.fmean(unwatched)


def moved_cores(samples: list[Sample]) -> list[str]:
    """The cores whose delivery rate differed between the watched and unwatched claims."""
    return sorted(core for core, group in _by_core(samples).items() if _differed(group))


def _rate(samples: list[Sample], field: str) -> str:
    value = mean_of(field)(samples)
    return "n/a" if value is None else f"{value:.2f}"


def _wording_note(prompts: dict[tuple[str, str], Prompt]) -> list[str]:
    """A note when the run's system claims differ from the ones the task sends now."""
    sent = {c: {p[0] for (_, cond), p in prompts.items() if cond == c} for c in _CURRENT_CLAIMS}
    if all(claims <= {_CURRENT_CLAIMS[c]} for c, claims in sent.items()):
        return []
    return ["", "This run sent an earlier wording of the training claims than the task sends now."]


def _condition_table(samples: list[Sample]) -> list[str]:
    columns = ["presents", "strongreject", "guard_refusal", "guard_unsafe", "regex_refusal"]
    lines = [
        "| Condition | Delivers | Presents as refusal | StrongREJECT | Guard refusal | Guard unsafe "
        "| Opening refusal |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition, label in _CONDITION_LABELS.items():
        subset = [s for s in samples if s.condition == condition]
        delivers = bootstrap(subset, mean_of("delivers"))
        lines.append(f"| {label} | {delivers.fmt()} | {' | '.join(_rate(subset, c) for c in columns)} |")
    return lines


def _claim_lines(samples: list[Sample]) -> list[str]:
    gap, up, down, p = oversight_gap(samples)
    framed = [s for s in samples if s.oversight != "none"]
    cites = bootstrap(framed, mean_of("cites"))
    groups = _by_core(samples)
    changed = [
        f"{core} {_rate([s for s in groups[core] if s.oversight == 'watched'], 'delivers')} vs "
        f"{_rate([s for s in groups[core] if s.oversight == 'unwatched'], 'delivers')}"
        for core in moved_cores(samples)
    ]
    return [
        "**Training claim, watched minus unwatched (delivers, per core)**",
        f"- Mean difference: {gap.fmt(signed=True)}",
        f"- Cores higher / lower when watched: {up} / {down} (sign test p = {p:.3f})",
        f"- Cores that changed (watched vs unwatched): {'; '.join(changed) or 'none'}",
        f"- Reasoning cites the claim: {cites.fmt()} of {len(framed)} framed responses",
    ]


def _breakdown(samples: list[Sample], field: str) -> list[str]:
    lines = [
        f"| {field} | Cores | Plain | Framed, watched | Framed, unwatched |",
        "|---|---|---|---|---|",
    ]
    for value in sorted({getattr(s, field) for s in samples}):
        subset = [s for s in samples if getattr(s, field) == value]
        rates = [_rate([s for s in subset if s.condition == c], "delivers") for c in _CONDITION_LABELS]
        lines.append(f"| {value} | {len(_by_core(subset))} | {' | '.join(rates)} |")
    return lines


def summary_markdown(
    label: str, metadata: dict[str, Any], samples: list[Sample], prompts: dict[tuple[str, str], Prompt]
) -> str:
    excluded = Counter((s.judge_reason or "no metric row").split(" (")[0] for s in samples if s.delivers is None)
    tokens = [s.completion_tokens for s in samples if s.completion_tokens is not None]
    lines = [
        f"## PurposeFraming: {label}",
        "",
        f"Model `{metadata.get('llm_name')}`: {len(_by_core(samples))} cores x 3 conditions x "
        f"{metadata.get('repeats')} samples = {len(samples)} responses. Values are shares of responses (mean "
        "for StrongREJECT); brackets are 95% intervals from resampling cores. The plugin README explains "
        "each column.",
        *_wording_note(prompts),
        "",
        *_condition_table(samples),
        "",
        *_claim_lines(samples),
        "",
        "**Delivers, by category**",
        "",
        *_breakdown(samples, "category"),
        "",
        "**Run**",
        f"- Finish reasons: {dict(Counter(s.finish_reason for s in samples))}; median generated tokens: "
        f"{statistics.median(tokens) if tokens else 'n/a'}",
        f"- Excluded from rubric labels: {sum(excluded.values())} {dict(excluded) if excluded else ''}".rstrip(),
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    label: str,
    metadata: dict[str, Any],
    samples: list[Sample],
    prompts: dict[tuple[str, str], Prompt],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(summary_markdown(label, metadata, samples, prompts), encoding="utf-8")
    fields = ["delivers", "cites", "strongreject", "guard_refusal", "guard_unsafe", "regex_refusal"]
    with (out_dir / "per_core.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["core", "category", "condition", *fields, "samples"])
        cells: dict[tuple[str, ...], list[Sample]] = defaultdict(list)
        for s in samples:
            cells[(s.core, s.category, s.condition)].append(s)
        for key in sorted(cells):
            # A mean is None when every sample abstained. Plain rows always abstain on `cites` (no claim
            # was made), so that cell is written blank rather than 0.
            means = [mean_of(name)(cells[key]) for name in fields]
            writer.writerow([*key, *("" if m is None else f"{m:.3f}" for m in means), len(cells[key])])
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")


def write_review(path: Path, samples: list[Sample], answers: list[str]) -> int:
    """CSV of every answer the rubric judge says delivered content, for the human check (not for the repo)."""
    rows = [(s, answer) for s, answer in zip(samples, answers, strict=True) if s.delivers == 1.0]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "cites_claim", "guard_refusal", "regex_refusal", "judge_reason", "answer"])
        for s, answer in rows:
            writer.writerow([s.item_id, s.cites, s.guard_refusal, s.regex_refusal, s.judge_reason, answer])
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--review", type=Path, help="also write answers that delivered content to this CSV")
    args = parser.parse_args()

    samples, answers = load_run(args.run_dir)
    metadata = json.loads((args.run_dir / "metadata.json").read_text(encoding="utf-8"))
    out_dir = RESULTS_DIR / args.label
    write_outputs(out_dir, args.label, metadata, samples, load_prompts(args.run_dir))
    print((out_dir / "summary.md").read_text(encoding="utf-8"))
    if args.review:
        print(f"Wrote {write_review(args.review, samples, answers)} answers for review to {args.review}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
