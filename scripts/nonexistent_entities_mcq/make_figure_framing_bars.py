#!/usr/bin/env python3
r"""Bar chart comparing the prompt framings at the last checkpoints, immediate vs reasoned answers.

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figure_framing_bars.py \
        --summary <sweep summary.json> --reasoned <reasoned_summary.json> \
        --metadata <prompt_metadata.jsonl from the NonexistentEntities task> --out <png>

One column per checkpoint (RL 100, RL 400, Think), framings on the x-axis, two bars per framing: the
lettered prompt with the IDK option answered immediately and after a reasoning trace. Top row accuracy
on the fake items of that framing, bottom row abstention. Error bars are 95% bootstrap intervals over
the framing's items (10 items, 5 for half-remembered).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figure_prompt_framings import IMMEDIATE, IMMEDIATE_TASK, REASONED, REASONED_TASK, _bootstrap  # noqa: E402
from make_figures import INK, INK2, style  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FRAMINGS = ["authority", "direct", "comparison", "online", "half-remembered"]
CHECKPOINTS = [
    ("allenai_Olmo-3-32B-Think@step_100", "RL step 100"),
    ("allenai_Olmo-3-32B-Think@step_400", "RL step 400"),
    ("allenai_Olmo-3-32B-Think@main", "Think (final)"),
]


def _stat(summary: dict, task: str, label: str, key: str, items: list[str]) -> tuple[float, float, float] | None:
    per_item = summary.get(task, {}).get(label, {}).get("main", {}).get(key)
    if not per_item:
        return None
    values = [per_item[k] for k in items if k in per_item]
    if not values:
        return None
    mean = sum(values) / len(values)
    lo, hi = _bootstrap(values)
    return mean, max(0.0, mean - lo), max(0.0, hi - mean)


def draw(immediate: dict, reasoned: dict, framing_of: dict[str, str], out: Path) -> None:
    fig, axes = plt.subplots(2, len(CHECKPOINTS), figsize=(13.5, 6.6), dpi=160, sharex=True, sharey="row")
    width = 0.36
    for col, (label, name) in enumerate(CHECKPOINTS):
        for row, (key, ylabel, ylim) in enumerate(
            (("item_accuracy", "accuracy (all option orders)", 1.0), ("item_abstention", "abstention rate", 0.3))
        ):
            ax = axes[row, col]
            for i, framing in enumerate(FRAMINGS):
                items = sorted(k for k, f in framing_of.items() if f == framing)
                for dx, source, task, color in (
                    (-width / 2, immediate, IMMEDIATE_TASK, IMMEDIATE),
                    (width / 2, reasoned, REASONED_TASK, REASONED),
                ):
                    s = _stat(source, task, label, key, items)
                    if s is None:
                        continue
                    mean, lo, hi = s
                    ax.bar(i + dx, mean, width=width - 0.03, color=color, zorder=2)
                    ax.errorbar(i + dx, mean, yerr=[[lo], [hi]], color=INK2, lw=0.9, capsize=2.5, zorder=3, alpha=0.8)
            ax.set_ylim(0, ylim)
            ax.set_xlim(-0.6, len(FRAMINGS) - 0.4)
            if row == 0:
                ax.axhline(0.20, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
                ax.set_title(name, fontsize=10.5, loc="left", color=INK)
            if col == 0:
                ax.set_ylabel(ylabel)
            style(ax)
            if row == 1:
                ax.set_xticks(
                    range(len(FRAMINGS)),
                    [f"{f}\n({sum(1 for v in framing_of.values() if v == f)} items)" for f in FRAMINGS],
                    fontsize=8,
                )
    axes[0, 0].text(-0.55, 0.205, "chance", fontsize=6.5, color=INK2, ha="left", va="bottom")
    handles = [
        Patch(color=IMMEDIATE, label="immediate answer (letters + “I don't know”)"),
        Patch(color=REASONED, label="after reasoning (thinking), same prompt"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Fake entities by prompt framing at the last three checkpoints: "
        "accuracy (picks “does not exist”) and abstention",
        fontsize=11.5,
        color=INK,
        x=0.05,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.99,
        0.006,
        "fake items only (the 5 controls are all framed directly); error bars: 95% bootstrap over the framing's items",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.865, bottom=0.1, wspace=0.08, hspace=0.12)
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
