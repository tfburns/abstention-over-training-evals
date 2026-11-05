#!/usr/bin/env python3
"""Draw the figures for the NonexistentEntities PR from the hand labels.

    uv run --with matplotlib python scripts/nonexistent_entities/make_figures.py

Reads ``labels/hand_labels.csv`` (id, sample, verdict) and ``prompt_metadata.jsonl`` and writes two
PNGs to ``figures/``: fabrication rate per domain (with Wilson 95 % intervals) and the distribution of
per-prompt fabrication counts out of 5 samples. Pass ``--labels`` to draw the same figures from
another label file (for example the judge's verdicts).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DOMAINS = {
    "pharma": "Prescription drugs",
    "peds": "Paediatric dosing",
    "proc": "Procedures & scores",
    "osha": "Workplace safety",
    "elec": "Electrical code",
}
# Fixed categorical order (one hue per domain, never cycled); blue is also the single-series hue.
COLORS = {"pharma": "#2a78d6", "peds": "#eb6834", "proc": "#1baf7a", "osha": "#eda100", "elec": "#e87ba4"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def load(labels: Path) -> list[dict[str, object]]:
    meta = {m["id"]: m for m in (json.loads(x) for x in (HERE / "prompt_metadata.jsonl").read_text().splitlines() if x)}
    rows = []
    for r in csv.DictReader(labels.open(encoding="utf-8")):
        v = r["verdict"].strip().upper()
        if v not in ("UNSAFE", "SAFE") or r["id"].startswith("ctrl-"):
            continue
        m = meta[r["id"]]
        rows.append({"id": r["id"], "sample": int(r["sample"]), "unsafe": v == "UNSAFE",
                     "domain": m["domain"], "framing": m["framing"]})
    return rows


def fig_domain(rows: list[dict[str, object]], out: Path, source: str) -> None:
    stats = []
    for d, name in DOMAINS.items():
        sub = [r for r in rows if r["domain"] == d]
        k, n = sum(1 for r in sub if r["unsafe"]), len(sub)
        stats.append((name, k, n, *wilson(k, n)))
    stats.sort(key=lambda s: s[1] / s[2])
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=160)
    y = range(len(stats))
    rates = [100 * s[1] / s[2] for s in stats]
    ax.barh(list(y), rates, height=0.55, color="#2a78d6", zorder=2)
    for i, s in enumerate(stats):
        ax.plot([100 * s[3], 100 * s[4]], [i, i], color=INK, lw=1.2, zorder=3)
        ax.plot([100 * s[3]] * 2, [i - 0.12, i + 0.12], color=INK, lw=1.2, zorder=3)
        ax.plot([100 * s[4]] * 2, [i - 0.12, i + 0.12], color=INK, lw=1.2, zorder=3)
        ax.text(100 * s[4] + 1.5, i, f"{s[1]}/{s[2]}  ({rates[i]:.0f}%)", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [s[0] for s in stats])
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("answers that fabricate specifics about the non-existent entity (%); lines = Wilson 95% CI")
    ax.grid(axis="x", color=GRID, zorder=0)
    all_k, all_n = sum(1 for r in rows if r["unsafe"]), len(rows)
    ax.set_title(f"Olmo-3-32B-Think fabrication rate by domain ({source}; overall {all_k}/{all_n} = "
                 f"{100 * all_k / all_n:.0f}%)", fontsize=10, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_per_prompt(rows: list[dict[str, object]], out: Path, source: str) -> None:
    per = defaultdict(int)
    dom_of = {}
    for r in rows:
        per[r["id"]] += r["unsafe"]
        dom_of[r["id"]] = r["domain"]
    counts = {d: Counter() for d in DOMAINS}
    for pid, k in per.items():
        counts[dom_of[pid]][k] += 1
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    bottom = [0] * 6
    for d, name in DOMAINS.items():
        vals = [counts[d][k] for k in range(6)]
        ax.bar(range(6), vals, bottom=bottom, width=0.62, color=COLORS[d], label=name, zorder=2,
               edgecolor="white", linewidth=1.5)
        for k in range(6):
            if vals[k]:
                ax.text(k, bottom[k] + vals[k] / 2, str(vals[k]), ha="center", va="center", fontsize=8, color=INK)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(6), [str(k) for k in range(6)])
    ax.set_xlabel("fabricated answers out of 5 samples of the same prompt")
    ax.set_ylabel("prompts")
    ax.set_title(f"How consistent is the failure per prompt? ({len(per)} prompts, {source})", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, title="domain", title_fontsize=8)
    ax.grid(axis="y", color=GRID, zorder=0)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=HERE / "labels" / "hand_labels.csv")
    ap.add_argument("--out-dir", type=Path, default=HERE / "figures")
    ap.add_argument("--source", default="hand labels", help="how the labels were made, for the titles")
    ap.add_argument("--suffix", default="", help="appended to the file names, e.g. _judge")
    args = ap.parse_args()
    rows = load(args.labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_domain(rows, args.out_dir / f"fig1_rate_by_domain{args.suffix}.png", args.source)
    fig_per_prompt(rows, args.out_dir / f"fig2_per_prompt_counts{args.suffix}.png", args.source)
    print(f"{len(rows)} labelled answers; wrote 2 figures to {args.out_dir}")


if __name__ == "__main__":
    main()
