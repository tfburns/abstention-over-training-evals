#!/usr/bin/env python3
r"""Figures for the reasoned MCQ runs: a reasoned-only set and a comparison set against the sweep.

    uv run --with matplotlib python scripts/nonexistent_entities_mcq/make_figures_reasoned.py \
        --reasoned reasoned_summary.json --immediate summary.json --out-dir figures

Reasoned-only set (``r1``..``r3``): accuracy, abstention (IDK task, with the immediate sweep's
abstention overlaid) and answers without a letter per checkpoint; median trace length per outcome
(answer text excluded, traces cut at the cap excluded); per-category accuracy of the final Think
model. Comparison set (``c1``..``c2``): immediate (log-likelihood) versus reasoned accuracy on the
same checkpoints, with the paired per-item difference; a per-item scatter for the final Think model.
Error bars are 95% bootstrap intervals over items.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = [
    ("allenai_Olmo-3-32B-Think-SFT@5e-5-step5000", "SFT 5k"),
    ("allenai_Olmo-3-32B-Think-SFT@main", "SFT final"),
    ("allenai_Olmo-3-32B-Think-DPO@main", "DPO"),
    ("allenai_Olmo-3-32B-Think@step_100", "RL 100"),
    ("allenai_Olmo-3-32B-Think@step_400", "RL 400"),
    ("allenai_Olmo-3-32B-Think@main", "Think"),
]
# (reasoned task, paired immediate task, name, colour, marker, chance)
TASKS = [
    ("NonexistentEntitiesMCQReasoned", "NonexistentEntitiesMCQ", "4 options", "#2a78d6", "o", 0.25),
    ("NonexistentEntitiesMCQReasonedIdk", "NonexistentEntitiesMCQIdk", "5 options with IDK", "#eb6834", "s", 0.20),
]
CATEGORIES = {
    "pharma": "Prescription drugs",
    "peds": "Paediatric dosing",
    "proc": "Procedures & scores",
    "osha": "Workplace safety",
    "elec": "Electrical code",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
IMMEDIATE_COLOR = "#888780"
THINK = "allenai_Olmo-3-32B-Think@main"
IDK_QUOTE = "share answering “I don't know”"


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


def _xaxis(ax: plt.Axes) -> None:
    ax.set_xticks(range(len(ORDER)), [n for _, n in ORDER], rotation=25, ha="right")
    ax.set_xlim(-0.5, len(ORDER) - 0.5)


def _series(summary: dict, task: str, subject: str, key: str, ci_key: str | None):
    xs, ys, lo, hi = [], [], [], []
    for i, (label, _) in enumerate(ORDER):
        s = summary.get(task, {}).get(label, {}).get(subject)
        if not s or s.get(key) is None:
            continue
        xs.append(i)
        ys.append(s[key])
        ci = s.get(ci_key) if ci_key else None
        lo.append(max(0.0, s[key] - ci[0]) if ci else 0.0)
        hi.append(max(0.0, ci[1] - s[key]) if ci else 0.0)
    return xs, ys, [lo, hi]


def _plot(ax, xs, ys, err, **kw):
    ax.errorbar(xs, ys, yerr=err, lw=2, ms=7, mec="white", mew=1.2, capsize=2.5, elinewidth=0.9, zorder=3, **kw)


def fig_r1(reasoned: dict, out: Path, immediate: dict | None = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=160)
    for task, imm_task, name, color, marker, _ in TASKS:
        for subject, ls, suffix in (("main", "-", ", fake"), ("control", (0, (4, 2)), ", controls")):
            xs, ys, err = _series(reasoned, task, subject, "accuracy", "accuracy_ci95")
            if xs:
                _plot(axes[0], xs, ys, err, color=color, ls=ls, marker=marker, label=name + suffix)
            if task.endswith("Idk"):
                xs, ys, err = _series(reasoned, task, subject, "abstention_rate", None)
                if xs:
                    _plot(axes[1], xs, ys, err, color=color, ls=ls, marker=marker, label=f"reasoned{suffix}")
                if immediate:
                    xi, yi, ei = _series(immediate, imm_task, subject, "abstention_rate", None)
                    if xi:
                        _plot(
                            axes[1], xi, yi, ei, color=IMMEDIATE_COLOR, ls=ls, marker=marker, label=f"immediate{suffix}"
                        )
        xs, ys, err = _series(reasoned, task, "main", "unparseable_rate", None)
        if xs:
            _plot(axes[2], xs, ys, err, color=color, ls="-", marker=marker, label=f"{name}: no letter (incl. cut)")
        xs, ys, err = _series(reasoned, task, "main", "truncated_rate", None)
        if xs:
            _plot(axes[2], xs, ys, err, color=color, ls=(0, (1, 2)), marker=marker, label=f"{name}: cut at 16k tokens")
    for ax, title, ylabel, top in (
        (axes[0], "Accuracy with reasoning", "share of answers on the gold letter", 1.0),
        (axes[1], "Abstention, IDK task: reasoned vs immediate", IDK_QUOTE, 0.12),
        (axes[2], "Fake items: answers without a letter", "share", 0.12),
    ):
        _xaxis(ax)
        ax.set_ylim(0, top)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(ylabel)
        style(ax)
    axes[0].axhline(0.25, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right", handlelength=3)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right", handlelength=3)
    axes[2].legend(frameon=False, fontsize=7, loc="upper right", handlelength=3)
    axes[2].text(
        0.02, 0.92, "SFT 5k: prose answers with no letter", transform=axes[2].transAxes, fontsize=6, color=INK2
    )
    fig.text(0.99, 0.005, "error bars: 95% bootstrap over items", fontsize=7, color=INK2, ha="right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_r2(reasoned: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
    for task, _, name, color, marker, _ in TASKS:
        xs, ok, _ = _series(reasoned, task, "main", "trace_tokens_median_correct", None)
        xs2, bad, _ = _series(reasoned, task, "main", "trace_tokens_median_wrong", None)
        if xs:
            ax.plot(
                xs,
                ok,
                color=color,
                lw=2,
                marker=marker,
                ms=7,
                mec="white",
                mew=1.2,
                label=f"{name}: correct answers",
                zorder=3,
            )
        if xs2:
            ax.plot(
                xs2,
                bad,
                color=color,
                lw=2,
                ls=(0, (4, 2)),
                marker=marker,
                ms=7,
                mec="white",
                mew=1.2,
                label=f"{name}: wrong answers",
                zorder=3,
            )
    _xaxis(ax)
    ax.set_ylim(0, 3800)
    ax.set_ylabel("median trace tokens (fake items)")
    ax.set_title(
        "How long the model reasons before answering: median trace length per outcome", fontsize=10, loc="left"
    )
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left", handlelength=3)
    fig.text(
        0.99,
        0.01,
        "answer text excluded (4 chars per token); traces cut at the 16k cap excluded",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _category_means(items: dict[str, float]) -> list[float]:
    values = []
    for cat in CATEGORIES:
        accs = [v for k, v in items.items() if k.startswith(cat + "-")]
        values.append(sum(accs) / len(accs) if accs else 0.0)
    return values


def fig_r3(reasoned: dict, immediate: dict | None, out: Path) -> None:
    """Per-category accuracy of the final Think model, immediate answer vs reasoned (4-option task)."""
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=160)
    task, imm_task, _, color, _, _ = TASKS[0]
    series = []
    i = immediate.get(imm_task, {}).get(THINK, {}).get("main") if immediate else None
    r = reasoned.get(task, {}).get(THINK, {}).get("main")
    if i:
        series.append(("immediate answer", IMMEDIATE_COLOR, _category_means(i["item_accuracy"])))
    if r:
        series.append(("reasoned answer", color, _category_means(r["item_accuracy"])))
    width = 0.38
    for j, (sname, scolor, values) in enumerate(series):
        xs = [k + (j - (len(series) - 1) / 2) * width for k in range(len(CATEGORIES))]
        ax.bar(xs, values, width=width, color=scolor, label=sname, zorder=2)
        for x, v in zip(xs, values):
            ax.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(CATEGORIES)), list(CATEGORIES.values()), fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy, fake items")
    ax.set_title(
        "Think final, 4 options: accuracy by category, immediate vs reasoned (9 fake items each)",
        fontsize=10,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_c1(reasoned: dict, immediate: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), dpi=160, sharex=True)
    for ax, subject, title in (
        (axes[0], "main", "Fake entities: immediate answer vs reasoned answer (45 items)"),
        (axes[1], "control", "Real entities (controls): immediate vs reasoned (5 items)"),
    ):
        for task, imm_task, name, color, marker, chance in TASKS:
            xs, ys, err = _series(reasoned, task, subject, "accuracy", "accuracy_ci95")
            if not xs:
                continue  # no reasoned run for this task yet: nothing to pair
            _plot(ax, xs, ys, err, color=color, marker=marker, label=f"{name}, reasoned")
            xs2, ys2, err2 = _series(immediate, imm_task, subject, "accuracy_all_orders", "accuracy_ci95")
            if xs2:
                _plot(
                    ax,
                    xs2,
                    ys2,
                    err2,
                    color=color,
                    marker=marker,
                    ls=(0, (4, 2)),
                    alpha=0.55,
                    label=f"{name}, immediate",
                )
            ax.axhline(chance, color=color, lw=0.8, ls=(0, (3, 3)), zorder=1, alpha=0.6)
            ax.text(len(ORDER) - 0.55, chance + 0.008, f"chance, {name}", fontsize=6, color=color, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy over all option orders")
        ax.set_title(title, fontsize=10, loc="left")
        style(ax)
    _xaxis(axes[1])
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="lower right", handlelength=3)
    fig.text(
        0.99,
        0.005,
        "error bars: 95% bootstrap over items; dashed = immediate (log-likelihood), solid = reasoned",
        fontsize=7,
        color=INK2,
        ha="right",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_c2(reasoned: dict, immediate: dict, out: Path) -> None:
    """Paired per-item difference (reasoned minus immediate) per checkpoint, and the Think scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=160, gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    for task, _, name, color, marker, _ in TASKS:
        xs, ys, err = _series(reasoned, task, "main", "paired_difference", "paired_difference_ci95")
        if xs:
            _plot(ax, xs, ys, err, color=color, marker=marker, label=name)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    _xaxis(ax)
    ax.set_ylim(-0.2, 0.4)
    ax.set_ylabel("reasoned minus immediate (fake items)")
    ax.set_title("Paired gain from reasoning, same items and orders", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    ax = axes[1]
    for task, imm_task, name, color, marker, _ in TASKS:
        r = reasoned.get(task, {}).get(THINK, {}).get("main")
        i = immediate.get(imm_task, {}).get(THINK, {}).get("main")
        if not r or not i:
            continue
        shared = [k for k in r["item_accuracy"] if k in i.get("item_accuracy", {})]
        counts = Counter((round(i["item_accuracy"][k], 2), round(r["item_accuracy"][k], 2)) for k in shared)
        ax.scatter(
            [x for x, _ in counts],
            [y for _, y in counts],
            s=[18 + 14 * c for c in counts.values()],
            color=color,
            marker=marker,
            alpha=0.75,
            label=f"{name} (size = number of items)",
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        if task == TASKS[0][0]:  # label outliers once, on the 4-option task
            for k in shared:
                x, y = i["item_accuracy"][k], r["item_accuracy"][k]
                if y - x < -0.15 or (x < 0.5 and y < 0.6):
                    ax.annotate(k, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=6, color=INK2)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("immediate answer: share of orders correct")
    ax.set_ylabel("reasoned answer: share of orders correct")
    ax.set_title("Think final, per fake item", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    style(ax)
    ax.grid(axis="x", color=GRID, zorder=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reasoned", type=Path, required=True, help="reasoned_summary.json")
    ap.add_argument("--immediate", type=Path, default=None, help="the sweep's summary.json, for the comparison set")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    reasoned = json.loads(args.reasoned.read_text(encoding="utf-8"))
    immediate = json.loads(args.immediate.read_text(encoding="utf-8")) if args.immediate else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_r1(reasoned, args.out_dir / "r1_reasoned_accuracy_abstention.png", immediate)
    fig_r2(reasoned, args.out_dir / "r2_reasoning_length.png")
    fig_r3(reasoned, immediate, args.out_dir / "r3_think_by_category.png")
    written = 3
    if immediate:
        fig_c1(reasoned, immediate, args.out_dir / "c1_immediate_vs_reasoned.png")
        fig_c2(reasoned, immediate, args.out_dir / "c2_paired_difference.png")
        written += 2
    print(f"wrote {written} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
