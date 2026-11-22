#!/usr/bin/env python3
r"""Accuracy and abstention per prompt framing (direct, authority, comparison, online, half-remembered).

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figure_prompt_framings.py \
        --summary <sweep summary.json> --reasoned <reasoned_summary.json> \
        --metadata <prompt_metadata.jsonl from the NonexistentEntities task> --out <png>

The framing of each item comes from the generative task's ``prompt_metadata.jsonl`` (same item ids).
One column per framing, fake items only (the five controls are all framed directly). Lines: the
lettered prompt with the IDK option answered immediately (13 checkpoints) and after a reasoning trace
(post-training checkpoints); the grey dashed line is the average over all 45 fake items for reference.
Error bars are 95% bootstrap intervals over the items of that framing (10 items, 5 for half-remembered).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import INK, INK2, ORDER, POST_START, style  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

FRAMINGS = ["direct", "authority", "comparison", "online", "half-remembered"]
IMMEDIATE, REASONED, ALL = "#2a78d6", "#eb6834", "#9a9893"
IMMEDIATE_TASK, REASONED_TASK = "NonexistentEntitiesMCQIdk", "NonexistentEntitiesMCQReasonedIdk"


def _bootstrap(values: list[float], resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(resamples))
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def _series(summary: dict, task: str, item_key: str, items: list[str] | None):
    """Per checkpoint: mean over the given items (all fake items if None) with a bootstrap interval."""
    xs, ys, lo, hi = [], [], [], []
    for i, (label, _) in enumerate(ORDER):
        per_item = summary.get(task, {}).get(label, {}).get("main", {}).get(item_key)
        if not per_item:
            continue
        values = [per_item[k] for k in (items or sorted(per_item)) if k in per_item]
        if not values:
            continue
        mean = sum(values) / len(values)
        a, b = _bootstrap(values)
        xs.append(i)
        ys.append(mean)
        lo.append(max(0.0, mean - a))
        hi.append(max(0.0, b - mean))
    return xs, ys, [lo, hi]


def _plot(ax, xs, ys, err, color, marker, dx):
    container = ax.errorbar(
        [x + dx for x in xs],
        ys,
        yerr=err,
        color=color,
        marker=marker,
        lw=1.8,
        ms=5.5,
        mec="white",
        mew=1.0,
        capsize=2,
        elinewidth=0.8,
        zorder=3,
    )
    for part in (*container[1], *container[2]):
        part.set_alpha(0.5)


def draw(immediate: dict, reasoned: dict, framing_of: dict[str, str], out: Path) -> None:
    fig, axes = plt.subplots(2, len(FRAMINGS), figsize=(15, 6.4), dpi=160, sharex=True, sharey="row")
    for col, framing in enumerate(FRAMINGS):
        items = sorted(k for k, f in framing_of.items() if f == framing)
        for row, (acc_key, ylabel, ylim) in enumerate(
            (("item_accuracy", "accuracy (all option orders)", 1.0), ("item_abstention", "abstention rate", 0.3))
        ):
            ax = axes[row, col]
            xs, ys, _ = _series(immediate, IMMEDIATE_TASK, acc_key, None)
            ax.plot(xs, ys, color=ALL, lw=1.2, ls=(0, (4, 2)), zorder=2)
            xs, ys, err = _series(immediate, IMMEDIATE_TASK, acc_key, items)
            _plot(ax, xs, ys, err, IMMEDIATE, "o", -0.08)
            xs, ys, err = _series(reasoned, REASONED_TASK, acc_key, items)
            if xs:
                _plot(ax, xs, ys, err, REASONED, "s", 0.08)
            ax.axvspan(POST_START - 0.5, len(ORDER) - 0.5, color="#f1efe8", zorder=0)
            ax.set_xlim(-0.5, len(ORDER) - 0.5)
            ax.set_ylim(0, ylim)
            if row == 0:
                ax.axhline(0.20, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
                ax.set_title(f"{framing} ({len(items)} items)", fontsize=10, loc="left", color=INK)
            if col == 0:
                ax.set_ylabel(ylabel)
            style(ax)
            ax.tick_params(labelsize=7)
            if row == 1:
                ax.set_xticks(range(len(ORDER)), [name for _, name in ORDER], rotation=50, ha="right")
    axes[0, 0].text(POST_START - 0.4, 0.98, "post-training", fontsize=7, color=INK2, va="top")
    axes[0, 0].text(-0.4, 0.98, "pre-training", fontsize=7, color=INK2, va="top")
    axes[0, 0].text(len(ORDER) - 0.6, 0.195, "chance", fontsize=6.5, color=INK2, ha="right", va="top")
    handles = [
        Line2D(
            [],
            [],
            color=IMMEDIATE,
            marker="o",
            lw=1.8,
            ms=5.5,
            mec="white",
            label="immediate answer, letters + “I don't know”",
        ),
        Line2D(
            [],
            [],
            color=REASONED,
            marker="s",
            lw=1.8,
            ms=5.5,
            mec="white",
            label="after reasoning, letters + “I don't know”",
        ),
        Line2D([], [], color=ALL, lw=1.2, ls=(0, (4, 2)), label="all 45 fake items, immediate answer"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Fake entities by prompt framing: accuracy (picks “does not exist”) and abstention over Olmo 3 32B training",
        fontsize=11.5,
        color=INK,
        x=0.045,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.99,
        0.005,
        "fake items only (the 5 controls are all framed directly); error bars: 95% bootstrap over the framing's items",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.86, bottom=0.13, wspace=0.08, hspace=0.12)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True, help="the immediate-answer sweep's summary.json")
    ap.add_argument("--reasoned", type=Path, required=True, help="reasoned_summary.json")
    ap.add_argument("--metadata", type=Path, required=True, help="prompt_metadata.jsonl (id, framing, control)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    immediate = json.loads(args.summary.read_text(encoding="utf-8"))
    reasoned = json.loads(args.reasoned.read_text(encoding="utf-8"))
    framing_of = {}
    for line in args.metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not row.get("control"):
                framing_of[row["id"]] = row["framing"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(immediate, reasoned, framing_of, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
