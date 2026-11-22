#!/usr/bin/env python3
r"""Draw the checkpoint-sweep figures from a summary.json written by ``summarise_runs.py``.

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figures.py \
        --summary outputs/nonexistent_entities_mcq/summary.json --out-dir figures

Checkpoints are placed in training order (stage 1 by step, stage 2, stage 3 = the released base, SFT,
DPO, RL = Think); checkpoints missing from the summary leave a gap. The cloze series use the
length-corrected (Bayesian) accuracy and abstention, the lettered series the raw argmax. Error bars
are the summariser's 95% bootstrap intervals over items.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Training order, label -> short name. Labels are summarise_runs.py checkpoint labels.
ORDER = [
    ("allenai_Olmo-3-1125-32B@stage1-step136000", "S1 136k"),
    ("allenai_Olmo-3-1125-32B@stage1-step270000", "S1 270k"),
    ("allenai_Olmo-3-1125-32B@stage1-step398000", "S1 398k"),
    ("allenai_Olmo-3-1125-32B@stage1-step528000", "S1 528k"),
    ("allenai_Olmo-3-1125-32B@stage1-step656000", "S1 656k"),
    ("allenai_Olmo-3-1125-32B@stage2-ingredient1+2-step23842", "S2 final"),
    ("allenai_Olmo-3-1125-32B@main", "base (S3)"),
    ("allenai_Olmo-3-32B-Think-SFT@5e-5-step5000", "SFT 5k"),
    ("allenai_Olmo-3-32B-Think-SFT@main", "SFT final"),
    ("allenai_Olmo-3-32B-Think-DPO@main", "DPO"),
    ("allenai_Olmo-3-32B-Think@step_100", "RL 100"),
    ("allenai_Olmo-3-32B-Think@step_400", "RL 400"),
    ("allenai_Olmo-3-32B-Think@main", "Think"),
]
# One hue per task, fixed order, never cycled; a marker shape per task as the second cue.
TASKS = [
    ("NonexistentEntitiesMCQ", "MCQ (letters)", "#2a78d6", "o", "accuracy_all_orders", None),
    ("NonexistentEntitiesMCQIdk", "MCQ + IDK", "#eb6834", "s", "accuracy_all_orders", "abstention_rate"),
    ("NonexistentEntitiesMCQCloze", "Cloze, corrected", "#1baf7a", "^", "accuracy_bayesian_recomputed", None),
    (
        "NonexistentEntitiesMCQClozeIdk",
        "Cloze + IDK, corrected",
        "#eda100",
        "D",
        "accuracy_bayesian_recomputed",
        "abstention_rate_bayesian",
    ),
]
# Interval key for each value key (95% bootstrap over items, from summarise_runs.py).
CI_KEYS = {
    "accuracy_all_orders": "accuracy_ci95",
    "accuracy_bayesian_recomputed": "accuracy_bayesian_ci95",
    "abstention_rate": "abstention_ci95",
    "abstention_rate_bayesian": "abstention_bayesian_ci95",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# Pre-training / post-training boundary sits after the base model.
POST_START = 7


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, zorder=0)


def _series(summary: dict, task: str, subject: str, key: str) -> tuple[list[int], list[float], list[list[float]]]:
    """X positions, values and asymmetric error-bar lengths (lower, upper) for one series."""
    xs, ys, lo, hi = [], [], [], []
    for i, (label, _) in enumerate(ORDER):
        stats = summary.get(task, {}).get(label, {}).get(subject, {})
        value = stats.get(key)
        if value is None:
            continue
        xs.append(i)
        ys.append(value)
        ci = stats.get(CI_KEYS.get(key, ""))
        lo.append(max(0.0, value - ci[0]) if ci else 0.0)
        hi.append(max(0.0, ci[1] - value) if ci else 0.0)
    return xs, ys, [lo, hi]


def _plot(ax: plt.Axes, xs: list[int], ys: list[float], err: list[list[float]], **kw: object) -> None:
    ax.errorbar(xs, ys, yerr=err, lw=2, ms=7, mec="white", mew=1.2, capsize=2.5, elinewidth=0.9, zorder=3, **kw)


def _axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_xticks(range(len(ORDER)), [name for _, name in ORDER], rotation=35, ha="right")
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.axvspan(POST_START - 0.5, len(ORDER) - 0.5, color="#f1efe8", zorder=0)
    ax.text(POST_START - 0.4, 0.98, "post-training", fontsize=8, color=INK2, va="top")
    ax.text(-0.4, 0.98, "pre-training", fontsize=8, color=INK2, va="top")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    style(ax)


def fig_accuracy(summary: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=160, sharex=True)
    for ax, subject, title in (
        (axes[0], "main", "Fake entities: share of prompts where the model picks “does not exist” (45 items)"),
        (
            axes[1],
            "control",
            "Real entities (controls): share of prompts where the model picks the true fact (5 items)",
        ),
    ):
        for task, name, color, marker, acc_key, _ in TASKS:
            xs, ys, err = _series(summary, task, subject, acc_key)
            if xs:
                _plot(ax, xs, ys, err, color=color, marker=marker, label=name)
        ax.axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.axhline(0.20, color=INK2, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.text(len(ORDER) - 0.6, 0.255, "chance, 4 options", fontsize=7, color=INK2, ha="right", va="bottom")
        ax.text(len(ORDER) - 0.6, 0.195, "chance, 5 options", fontsize=7, color=INK2, ha="right", va="top")
        _axis(ax, title, "accuracy over all option orders")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.88), ncol=2)
    fig.text(0.99, 0.005, "error bars: 95% bootstrap over items", fontsize=7, color=INK2, ha="right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_abstention(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=160)
    for task, name, color, marker, _, abst_key in TASKS:
        if abst_key is None:
            continue
        for subject, ls, suffix in (("main", "-", ", fake"), ("control", (0, (4, 2)), ", controls")):
            xs, ys, err = _series(summary, task, subject, abst_key)
            if xs:
                _plot(ax, xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
    _axis(ax, "How often “I don't know” is the top answer", "abstention rate")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.text(0.99, 0.01, "error bars: 95% bootstrap over items", fontsize=7, color=INK2, ha="right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_consistency(summary: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=160)
    for task, name, color, marker, _, _ in TASKS[:2]:
        for subject, ls, suffix in (("main", "-", ", fake"), ("control", (0, (4, 2)), ", controls")):
            xs, ys, err = _series(summary, task, subject, "consistency")
            if xs:
                _plot(ax, xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
    _axis(ax, "Share of items answered the same way under every option order", "consistency")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_accuracy(summary, args.out_dir / "fig1_accuracy_over_training.png")
    fig_abstention(summary, args.out_dir / "fig2_abstention_over_training.png")
    fig_consistency(summary, args.out_dir / "fig3_consistency_over_training.png")
    done = sum(1 for label, _ in ORDER if label in summary.get("NonexistentEntitiesMCQ", {}))
    print(f"{done} of {len(ORDER)} checkpoints in the summary; wrote 3 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
