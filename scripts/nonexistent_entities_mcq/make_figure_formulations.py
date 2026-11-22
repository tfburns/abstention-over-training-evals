#!/usr/bin/env python3
r"""One figure comparing the six question formulations on accuracy and abstention over training.

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figure_formulations.py \
        --summary <sweep summary.json> --reasoned <reasoned_summary.json> --out <png>

Formulations: the lettered multiple-choice prompt scored on the answer letter (immediate), the same
options scored as cloze continuations with the Bayesian length correction, and the lettered prompt
answered after a reasoning trace (post-training checkpoints only); each with and without the
"I don't know" option. Colour = how the question is asked and scored; dashed line and open marker =
the IDK option is present. Error bars are 95% bootstrap intervals over items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from make_figures import INK, INK2, ORDER, POST_START, style  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, GREEN, PURPLE = "#2a78d6", "#13996a", "#8b4fd6"
# (summary name, task, accuracy key, accuracy CI key, abstention key, abstention CI key, colour, marker, idk)
SERIES = [
    ("immediate", "NonexistentEntitiesMCQ", "accuracy_all_orders", "accuracy_ci95", None, None, BLUE, "o", False),
    (
        "immediate",
        "NonexistentEntitiesMCQIdk",
        "accuracy_all_orders",
        "accuracy_ci95",
        "abstention_rate",
        "abstention_ci95",
        BLUE,
        "o",
        True,
    ),
    (
        "immediate",
        "NonexistentEntitiesMCQCloze",
        "accuracy_bayesian_recomputed",
        "accuracy_bayesian_ci95",
        None,
        None,
        GREEN,
        "^",
        False,
    ),
    (
        "immediate",
        "NonexistentEntitiesMCQClozeIdk",
        "accuracy_bayesian_recomputed",
        "accuracy_bayesian_ci95",
        "abstention_rate_bayesian",
        "abstention_bayesian_ci95",
        GREEN,
        "^",
        True,
    ),
    ("reasoned", "NonexistentEntitiesMCQReasoned", "accuracy", "accuracy_ci95", None, None, PURPLE, "s", False),
    (
        "reasoned",
        "NonexistentEntitiesMCQReasonedIdk",
        "accuracy",
        "accuracy_ci95",
        "abstention_rate",
        "abstention_ci95",
        PURPLE,
        "s",
        True,
    ),
]
METHOD_NAMES = {BLUE: "letters, immediate answer", GREEN: "cloze, length-corrected", PURPLE: "letters, after reasoning"}


def _series(summary: dict, task: str, subject: str, key: str, ci_key: str | None):
    xs, ys, lo, hi = [], [], [], []
    for i, (label, _) in enumerate(ORDER):
        stats = summary.get(task, {}).get(label, {}).get(subject, {})
        value = stats.get(key)
        if value is None:
            continue
        xs.append(i)
        ys.append(value)
        ci = stats.get(ci_key) if ci_key else None
        lo.append(max(0.0, value - ci[0]) if ci else 0.0)
        hi.append(max(0.0, ci[1] - value) if ci else 0.0)
    return xs, ys, [lo, hi]


def _plot(ax, xs, ys, err, color, marker, idk, err_alpha=0.55):
    kw = dict(color=color, marker=marker, lw=1.8, ms=6.5, capsize=2.2, elinewidth=0.8, zorder=4 if idk else 3)
    if idk:
        kw.update(ls=(0, (4, 2)), mfc="white", mec=color, mew=1.6)
    else:
        kw.update(ls="-", mec="white", mew=1.0)
    container = ax.errorbar(xs, ys, yerr=err, **kw)
    for part in (*container[1], *container[2]):  # caps and bars: lighter than the line
        part.set_alpha(err_alpha)


def _xaxis(ax, top):
    ax.set_xticks(range(len(ORDER)), [name for _, name in ORDER], rotation=35, ha="right")
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.axvspan(POST_START - 0.5, len(ORDER) - 0.5, color="#f1efe8", zorder=0)
    if top:
        ax.text(
            POST_START - 0.4,
            0.985,
            "post-training",
            fontsize=8,
            color=INK2,
            va="top",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(-0.4, 0.985, "pre-training", fontsize=8, color=INK2, va="top", transform=ax.get_xaxis_transform())
    style(ax)


def draw(summaries: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 7.6),
        dpi=160,
        sharex="col",
        gridspec_kw={"height_ratios": [1.45, 1], "wspace": 0.16, "hspace": 0.12},
    )
    for col, subject in enumerate(("main", "control")):
        top, bottom = axes[0, col], axes[1, col]
        err_alpha = 0.55 if subject == "main" else 0.3  # five control items: intervals span most of the axis
        for source, task, acc, acc_ci, abst, abst_ci, color, marker, idk in SERIES:
            xs, ys, err = _series(summaries[source], task, subject, acc, acc_ci)
            if xs:
                _plot(top, xs, ys, err, color, marker, idk, err_alpha)
            if abst:
                xs, ys, err = _series(summaries[source], task, subject, abst, abst_ci)
                if xs:
                    _plot(bottom, xs, ys, err, color, marker, idk, err_alpha)
        top.axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        top.axhline(0.20, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
        cx, ha = (4.5, "center") if subject == "main" else (len(ORDER) - 0.6, "right")
        top.text(cx, 0.258, "chance, 4 options", fontsize=7, color=INK2, ha=ha, va="bottom", zorder=5)
        top.text(cx, 0.192, "chance, 5 options", fontsize=7, color=INK2, ha=ha, va="top", zorder=5)
        top.set_ylim(0, 1.0)
        bottom.set_ylim(0, 0.6)
        _xaxis(top, top=True)
        _xaxis(bottom, top=False)
        if col == 0:
            top.set_ylabel("accuracy (all option orders)")
            bottom.set_ylabel("abstention rate")
        else:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)
    axes[0, 0].set_title(
        "Accuracy · fake entities (45 items): picks “does not exist”", fontsize=10, loc="left", color=INK
    )
    axes[0, 1].set_title(
        "Accuracy · controls (5 real entities, wide intervals): picks the true fact", fontsize=10, loc="left", color=INK
    )
    axes[1, 0].set_title("Abstention · fake entities (framings with an IDK option)", fontsize=10, loc="left", color=INK)
    axes[1, 1].set_title("Abstention · controls (5 real entities)", fontsize=10, loc="left", color=INK)
    handles = []
    for color in (BLUE, GREEN, PURPLE):
        marker = {BLUE: "o", GREEN: "^", PURPLE: "s"}[color]
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                lw=1.8,
                ms=6.5,
                mec="white",
                label=f"{METHOD_NAMES[color]}, 4 options",
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                lw=1.8,
                ms=6.5,
                ls=(0, (4, 2)),
                mfc="white",
                mec=color,
                mew=1.6,
                label=f"{METHOD_NAMES[color]}, + “I don't know”",
            )
        )
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        "How the question formulation changes accuracy and abstention over Olmo 3 32B training",
        fontsize=11.5,
        color=INK,
        x=0.065,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.99,
        0.004,
        "solid = 4 options, dashed/open = with “I don't know”; cloze uses the Bayesian length correction; "
        "error bars: 95% bootstrap over items",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.87, bottom=0.11, wspace=0.1, hspace=0.22)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True, help="the immediate-answer sweep's summary.json")
    ap.add_argument("--reasoned", type=Path, required=True, help="reasoned_summary.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summaries = {
        "immediate": json.loads(args.summary.read_text(encoding="utf-8")),
        "reasoned": json.loads(args.reasoned.read_text(encoding="utf-8")),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    draw(summaries, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
